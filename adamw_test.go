package adamw

import (
	"math"
	"testing"
)

// ---------- helpers ----------

func almostEqual(a, b, absTol, relTol float64) bool {
	diff := math.Abs(a - b)
	if diff <= absTol {
		return true
	}
	scale := math.Max(math.Abs(a), math.Abs(b))
	return diff <= relTol*scale
}

func slicesAlmostEqual(a, b []float64, absTol, relTol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !almostEqual(a[i], b[i], absTol, relTol) {
			return false
		}
	}
	return true
}

func clone(x []float64) []float64 {
	y := make([]float64, len(x))
	copy(y, x)
	return y
}

// emulateOneStepManual performs one "manual" AdamW step
// with bias-correction and decoupled weight decay: matches the algorithm in the library.
type manualState struct {
	t        int64
	m, v     []float64
	pb1, pb2 float64 // β1^t, β2^t
}

func newManualState(dim int) *manualState {
	return &manualState{
		t:   0,
		m:   make([]float64, dim),
		v:   make([]float64, dim),
		pb1: 1.0,
		pb2: 1.0,
	}
}

func emulateOneStepManual(params, grad []float64, st *manualState,
	alpha, beta1, beta2, eps, eta, lambda float64) {

	st.t++
	st.pb1 *= beta1
	st.pb2 *= beta2
	bc1 := 1.0 - st.pb1
	bc2 := 1.0 - st.pb2

	for i := range params {
		g := grad[i]
		st.m[i] = beta1*st.m[i] + (1.0-beta1)*g
		st.v[i] = beta2*st.v[i] + (1.0-beta2)*g*g

		mhat := st.m[i] / bc1
		vhat := st.v[i] / bc2

		update := alpha * mhat / (math.Sqrt(vhat) + eps)
		params[i] -= eta*update + eta*alpha*lambda*params[i]
	}
}

// ---------- tests ----------

func TestAdamW_MatchesManualUpdateOverManySteps(t *testing.T) {
	t.Parallel()

	params := []float64{0.1, -0.2, 0.3}
	grad := []float64{0.01, -0.02, 0.03}

	alpha := 1e-3
	b1 := 0.9
	b2 := 0.999
	eps := 1e-8
	lambda := 1e-2

	opt, err := New(clone(params), Options{
		Alpha:       alpha,
		Beta1:       b1,
		Beta2:       b2,
		Eps:         eps,
		WeightDecay: lambda,
		Schedule:    NewFixedSchedule(1.0, 0), // η_t ≡ 1
	})
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	manualParams := clone(params)
	ms := newManualState(len(params))

	const steps = 500
	libParams := clone(params)

	for s := 0; s < steps; s++ {
		// library
		if err := opt.Step(libParams, grad); err != nil {
			t.Fatalf("Step error at %d: %v", s, err)
		}
		// manual reference
		emulateOneStepManual(manualParams, grad, ms, alpha, b1, b2, eps, 1.0, lambda)

		if !slicesAlmostEqual(libParams, manualParams, 1e-12, 1e-10) {
			t.Fatalf("mismatch at step %d:\nlib:    %#v\nmanual: %#v", s+1, libParams, manualParams)
		}
	}
}

func TestAdamW_ZeroGrad_PureDecoupledDecay(t *testing.T) {
	t.Parallel()

	params := []float64{1.0, -2.0, 0.5}
	grad := []float64{0.0, 0.0, 0.0}

	alpha := 1e-3
	b1 := 0.9
	b2 := 0.999
	eps := 1e-8
	lambda := 0.1
	eta := 1.0

	opt, err := New(clone(params), Options{
		Alpha:       alpha,
		Beta1:       b1,
		Beta2:       b2,
		Eps:         eps,
		WeightDecay: lambda,
		Schedule:    NewFixedSchedule(eta, 0),
	})
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	libParams := clone(params)
	const steps = 5
	for i := 0; i < steps; i++ {
		if err := opt.Step(libParams, grad); err != nil {
			t.Fatalf("Step error: %v", err)
		}
	}

	// Expectation: θ * (1 - η*α*λ)^steps
	f := math.Pow(1.0-eta*alpha*lambda, steps)
	want := []float64{
		params[0] * f,
		params[1] * f,
		params[2] * f,
	}
	if !slicesAlmostEqual(libParams, want, 1e-12, 1e-10) {
		t.Fatalf("pure decay mismatch:\ngot:  %#v\nwant: %#v", libParams, want)
	}
}

func TestAdamW_Normalization_NoRestarts(t *testing.T) {
	t.Parallel()

	// Normalization: λ = λ_norm * sqrt(b / (B * T))
	params := []float64{0.3, -0.6}
	grad := []float64{0, 0} // isolate only the effect of decay

	alpha := 1e-3
	b1 := 0.9
	b2 := 0.999
	eps := 1e-8

	lambdaNorm := 0.05
	batch := 4
	data := 100
	T := 10

	eta := 1.0

	opt, err := New(clone(params), Options{
		Alpha: alpha, Beta1: b1, Beta2: b2, Eps: eps,
		Norm: &NormConfig{
			LambdaNorm:    lambdaNorm,
			BatchSize:     batch,
			DatasetSize:   data,
			TotalEpochs:   T,
			StepsPerEpoch: 0, // no restarts — not needed
		},
		Schedule: NewFixedSchedule(eta, 0),
	})
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	const steps = 3
	libParams := clone(params)
	for i := 0; i < steps; i++ {
		if err := opt.Step(libParams, grad); err != nil {
			t.Fatalf("Step error: %v", err)
		}
	}

	lambda := lambdaNorm * math.Sqrt(float64(batch)/(float64(data)*float64(T)))
	f := 1.0
	for i := 0; i < steps; i++ {
		f *= (1.0 - eta*alpha*lambda)
	}
	want := []float64{params[0] * f, params[1] * f}
	if !slicesAlmostEqual(libParams, want, 1e-12, 1e-10) {
		t.Fatalf("normalized decay mismatch:\ngot:  %#v\nwant: %#v (lambda=%.12f)", libParams, want, lambda)
	}
}

func TestAdamWR_Normalization_WithWarmRestarts_Cosine(t *testing.T) {
	t.Parallel()

	// Setup: two periods: T0=20 steps (2 epochs), then T1=40 steps (4 epochs)
	// StepsPerEpoch=10 => T_i(in epochs)=periodSteps/10.
	params := []float64{2.0}
	grad := []float64{0.0} // isolate decay * with cosine η_t

	alpha := 1e-3
	b1 := 0.9
	b2 := 0.999
	eps := 1e-8
	lambdaNorm := 0.2
	batch := 32
	data := 10000
	stepsPerEpoch := 10

	initPeriod := 20 // steps
	tMult := 2.0     // => next period 40 steps

	sched1, err := NewCosineAnnealingWarmRestarts(initPeriod, tMult)
	if err != nil {
		t.Fatalf("schedule ctor error: %v", err)
	}

	opt, err := New(clone(params), Options{
		Alpha: alpha, Beta1: b1, Beta2: b2, Eps: eps,
		Norm: &NormConfig{
			LambdaNorm:    lambdaNorm,
			BatchSize:     batch,
			DatasetSize:   data,
			StepsPerEpoch: stepsPerEpoch, // important for converting periodSteps -> epochs
			TotalEpochs:   0,             // not used with restarts
		},
		Schedule: sched1,
	})
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// Expected value is calculated via an independent schedule instance,
	// repeating the exact discretization of Eta() before Tick() and PeriodInfo() (as in the optimizer).
	sched2, err := NewCosineAnnealingWarmRestarts(initPeriod, tMult)
	if err != nil {
		t.Fatalf("schedule ctor2 error: %v", err)
	}

	totalSteps := 60 // go through 20 + 40 steps
	libParams := clone(params)
	expected := params[0]

	for s := 0; s < totalSteps; s++ {
		// Library
		if err := opt.Step(libParams, grad); err != nil {
			t.Fatalf("Step error at %d: %v", s, err)
		}

		// Expectation calculation
		periodSteps, hasR := sched2.PeriodInfo()
		_ = hasR // it's true, just for symmetry
		Ti := float64(periodSteps) / float64(stepsPerEpoch)
		if Ti < 1.0 {
			Ti = 1.0
		}
		lambda := lambdaNorm * math.Sqrt(float64(batch)/(float64(data)*Ti))
		eta := sched2.Eta()
		expected *= (1.0 - eta*alpha*lambda) // zero gradient => pure decay with η_t

		sched2.Tick()
	}

	want := []float64{expected}
	if !slicesAlmostEqual(libParams, want, 1e-11, 1e-9) {
		t.Fatalf("AdamWR (cosine+norm) mismatch:\ngot:  %#v\nwant: %#v", libParams, want)
	}
}

func TestCosineAnnealingWarmRestarts_EtaAndPeriods(t *testing.T) {
	t.Parallel()

	s, err := NewCosineAnnealingWarmRestarts(10, 2.0)
	if err != nil {
		t.Fatalf("ctor error: %v", err)
	}
	// At the start of the period tcur=0 => η=1
	if eta := s.Eta(); !almostEqual(eta, 1.0, 0, 0) {
		t.Fatalf("eta at start: got %g want 1", eta)
	}
	// Half-period (after 5 ticks for 10 steps of the period) should give about 0.5
	for i := 0; i < 5; i++ {
		s.Tick()
	}
	etaMid := s.Eta()
	if !almostEqual(etaMid, 0.5, 1e-12, 1e-9) {
		t.Fatalf("eta mid-period: got %g want 0.5", etaMid)
	}
	// Go to the end of the period: after 5 more ticks — restart, period doubles.
	for i := 0; i < 5; i++ {
		s.Tick()
	}
	steps, hasR := s.PeriodInfo()
	if !hasR {
		t.Fatalf("expected restarts=true")
	}
	// After restart, period duration should become 20 (10*2)
	if steps != 20 {
		t.Fatalf("period after restart: got %d want 20", steps)
	}
	// And η starts again from 1
	if eta := s.Eta(); !almostEqual(eta, 1.0, 0, 0) {
		t.Fatalf("eta after restart: got %g want 1", eta)
	}
}

func TestResetState_Reproducibility(t *testing.T) {
	t.Parallel()

	params0 := []float64{0.5, -1.0, 2.0}
	grad := []float64{0.03, -0.01, 0.02}

	alpha := 1e-3
	b1, b2, eps := 0.9, 0.999, 1e-8
	lambda := 1e-2

	// Cosine schedule (to check that Reset() also resets it)
	s1, _ := NewCosineAnnealingWarmRestarts(8, 2.0)
	opt1, err := New(clone(params0), Options{
		Alpha: alpha, Beta1: b1, Beta2: b2, Eps: eps,
		WeightDecay: lambda,
		Schedule:    s1,
	})
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// First run
	paramsA := clone(params0)
	for i := 0; i < 25; i++ {
		if err := opt1.Step(paramsA, grad); err != nil {
			t.Fatalf("Step A error: %v", err)
		}
	}
	// Reset and second run from the same initial values
	opt1.ResetState()
	if opt1.CurrentStep() != 0 {
		t.Fatalf("CurrentStep after reset: got %d want 0", opt1.CurrentStep())
	}

	paramsB := clone(params0)
	for i := 0; i < 25; i++ {
		if err := opt1.Step(paramsB, grad); err != nil {
			t.Fatalf("Step B error: %v", err)
		}
	}
	if !slicesAlmostEqual(paramsA, paramsB, 1e-12, 1e-10) {
		t.Fatalf("reset reproducibility mismatch:\nA: %#v\nB: %#v", paramsA, paramsB)
	}
}

func TestConstructorAndStepErrors(t *testing.T) {
	t.Parallel()

	// Empty parameters
	if _, err := New([]float64{}, Options{}); err == nil {
		t.Fatalf("expected error on empty params")
	}

	// Invalid normalization
	if _, err := New([]float64{1}, Options{
		Norm: &NormConfig{
			LambdaNorm:  -0.1, // < 0
			BatchSize:   32,
			DatasetSize: 1000,
			TotalEpochs: 10,
		},
	}); err == nil {
		t.Fatalf("expected error on negative LambdaNorm")
	}

	if _, err := New([]float64{1}, Options{
		Norm: &NormConfig{
			LambdaNorm:  0.1,
			BatchSize:   0, // invalid
			DatasetSize: 1000,
			TotalEpochs: 10,
		},
	}); err == nil {
		t.Fatalf("expected error on zero BatchSize")
	}

	if _, err := New([]float64{1}, Options{
		Norm: &NormConfig{
			LambdaNorm:  0.1,
			BatchSize:   32,
			DatasetSize: 0, // invalid
			TotalEpochs: 10,
		},
	}); err == nil {
		t.Fatalf("expected error on zero DatasetSize")
	}

	// Invalid cosine schedule parameters
	if _, err := NewCosineAnnealingWarmRestarts(0, 2.0); err == nil {
		t.Fatalf("expected error on initialPeriodSteps<=0")
	}
	if _, err := NewCosineAnnealingWarmRestarts(10, 0.5); err == nil {
		t.Fatalf("expected error on tMult<1")
	}

	// Step error: different vector lengths
	opt, err := New([]float64{1, 2}, Options{Schedule: NewFixedSchedule(1, 0)})
	if err != nil {
		t.Fatalf("New error: %v", err)
	}
	if e := opt.Step([]float64{1, 2}, []float64{1}); e == nil {
		t.Fatalf("expected error on mismatched sizes")
	}

	// Step error: uninitialized optimizer
	var bad Optimizer // zero-value
	if e := bad.Step([]float64{1}, []float64{1}); e == nil {
		t.Fatalf("expected error on uninitialized optimizer")
	}
}

// ---------- benchmarks ----------

func BenchmarkStep_256(b *testing.B) {
	dim := 256
	params := make([]float64, dim)
	grad := make([]float64, dim)
	for i := 0; i < dim; i++ {
		params[i] = 0.1 * math.Sin(float64(i))
		grad[i] = 0.01 * math.Cos(float64(i))
	}
	opt, err := New(clone(params), Options{
		Alpha:       1e-3,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 1e-2,
		Schedule:    NewFixedSchedule(1.0, 0),
	})
	if err != nil {
		b.Fatalf("New error: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := opt.Step(params, grad); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStep_4096(b *testing.B) {
	dim := 4096
	params := make([]float64, dim)
	grad := make([]float64, dim)
	for i := 0; i < dim; i++ {
		params[i] = 0.1 * math.Sin(float64(i))
		grad[i] = 0.01 * math.Cos(float64(i))
	}
	opt, err := New(clone(params), Options{
		Alpha:       1e-3,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 1e-2,
		Schedule:    NewFixedSchedule(1.0, 0),
	})
	if err != nil {
		b.Fatalf("New error: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := opt.Step(params, grad); err != nil {
			b.Fatal(err)
		}
	}
}

func TestStep_ErrorOnParamLengthMismatch(t *testing.T) {
	t.Parallel()

	// Build optimizer for dim=3
	params0 := []float64{0.1, -0.2, 0.3}
	opt, err := New(clone(params0), Options{
		Alpha:    1e-3,
		Beta1:    0.9,
		Beta2:    0.999,
		Eps:      1e-8,
		Schedule: NewFixedSchedule(1.0, 0),
	})
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// Now attempt to step with a longer params/grad; must return an error.
	params := []float64{0.1, -0.2, 0.3, 0.4}
	grad := []float64{0.01, -0.02, 0.03, 0.04}
	if e := opt.Step(params, grad); e == nil {
		t.Fatalf("expected error on params length mismatch with internal state")
	}
}

func TestDecayMask_DisablesWeightDecayPerIndex(t *testing.T) {
	t.Parallel()

	params0 := []float64{1.0, 2.0, 3.0}
	grad := []float64{0.0, 0.0, 0.0} // isolate decay-only behavior

	mask := []bool{true, false, true} // disable decay for index 1

	alpha := 1e-3
	b1, b2, eps := 0.9, 0.999, 1e-8
	lambda := 0.1
	eta := 1.0

	opt, err := New(clone(params0), Options{
		Alpha:       alpha,
		Beta1:       b1,
		Beta2:       b2,
		Eps:         eps,
		WeightDecay: lambda,
		Schedule:    NewFixedSchedule(eta, 0),
		DecayMask:   mask,
	})
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	params := clone(params0)
	steps := 3
	for i := 0; i < steps; i++ {
		if err := opt.Step(params, grad); err != nil {
			t.Fatalf("Step error: %v", err)
		}
	}

	// For indices with decay enabled: θ *= (1 - η*α*λ)^steps
	f := math.Pow(1.0-eta*alpha*lambda, float64(steps))
	want := []float64{
		params0[0] * f, // decayed
		params0[1],     // NOT decayed due to mask=false
		params0[2] * f, // decayed
	}
	if !slicesAlmostEqual(params, want, 1e-12, 1e-10) {
		t.Fatalf("mask decay mismatch:\ngot:  %#v\nwant: %#v", params, want)
	}
}

// TestWeightDecayAlphaScaling_SanityCheck verifies the basic weight decay formula
// with alpha scaling: θ₁ = θ₀ * (1 - α*η*λ)
func TestWeightDecayAlphaScaling_SanityCheck(t *testing.T) {
	t.Parallel()

	// Sanity case: θ₀=1, g=0, λ=0.1, α=0.01, η=1 ⇒ θ₁=0.999
	params := []float64{1.0}
	grad := []float64{0.0} // zero gradient to isolate decay effect

	alpha := 0.01
	lambda := 0.1
	eta := 1.0

	opt, err := New(params, Options{
		Alpha:       alpha,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: lambda,
		Schedule:    NewFixedSchedule(eta, 0),
	})
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// Perform one step
	if err := opt.Step(params, grad); err != nil {
		t.Fatalf("Step error: %v", err)
	}

	// Expected: θ₁ = 1.0 * (1 - 0.01*1.0*0.1) = 1.0 * (1 - 0.001) = 0.999
	expected := 0.999
	if math.Abs(params[0]-expected) > 1e-12 {
		t.Fatalf("Alpha scaling sanity check failed:\ngot:      %.15f\nexpected: %.15f\ndiff:     %.2e",
			params[0], expected, math.Abs(params[0]-expected))
	}
}

// TestWeightDecayMask_SelectiveApplication verifies that DecayMask works correctly
// with alpha scaling: decay only where mask[i] == true
func TestWeightDecayMask_SelectiveApplication(t *testing.T) {
	t.Parallel()

	// DecayMask=[false,true], g=0 ⇒ decay only in second coordinate
	params := []float64{2.0, 3.0}
	grad := []float64{0.0, 0.0} // zero gradients
	mask := []bool{false, true} // decay disabled for first param, enabled for second

	alpha := 0.01
	lambda := 0.1
	eta := 1.0

	opt, err := New(params, Options{
		Alpha:       alpha,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: lambda,
		Schedule:    NewFixedSchedule(eta, 0),
		DecayMask:   mask,
	})
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	originalParams := []float64{params[0], params[1]}

	// Perform one step
	if err := opt.Step(params, grad); err != nil {
		t.Fatalf("Step error: %v", err)
	}

	// Expected results:
	// params[0]: no decay (mask=false) ⇒ should remain unchanged = 2.0
	// params[1]: with decay (mask=true) ⇒ 3.0 * (1 - 0.01*1.0*0.1) = 3.0 * 0.999 = 2.997
	expectedFirst := originalParams[0] // unchanged
	expectedSecond := originalParams[1] * (1.0 - alpha*eta*lambda)

	if math.Abs(params[0]-expectedFirst) > 1e-12 {
		t.Fatalf("Masked decay failed for index 0 (should be unchanged):\ngot:      %.15f\nexpected: %.15f",
			params[0], expectedFirst)
	}

	if math.Abs(params[1]-expectedSecond) > 1e-12 {
		t.Fatalf("Masked decay failed for index 1 (should be decayed):\ngot:      %.15f\nexpected: %.15f\ndiff:     %.2e",
			params[1], expectedSecond, math.Abs(params[1]-expectedSecond))
	}
}

// TestWeightDecayB1_TotalEpochsScaling verifies B.1 normalization:
// when T is doubled, λ decreases as 1/√2
func TestWeightDecayB1_TotalEpochsScaling(t *testing.T) {
	t.Parallel()

	// Test parameters for normalization effect isolation

	alpha := 0.001
	eta := 1.0
	lambdaNorm := 0.1
	batch := 32
	data := 1000

	// Test with T=10 epochs
	T1 := 10
	opt1, err := New([]float64{1.0}, Options{
		Alpha: alpha,
		Beta1: 0.9, Beta2: 0.999, Eps: 1e-8,
		Norm: &NormConfig{
			LambdaNorm:  lambdaNorm,
			BatchSize:   batch,
			DatasetSize: data,
			TotalEpochs: T1,
		},
		Schedule: NewFixedSchedule(eta, 0),
	})
	if err != nil {
		t.Fatalf("New opt1 error: %v", err)
	}

	// Test with T=20 epochs (doubled)
	T2 := 20
	opt2, err := New([]float64{1.0}, Options{
		Alpha: alpha,
		Beta1: 0.9, Beta2: 0.999, Eps: 1e-8,
		Norm: &NormConfig{
			LambdaNorm:  lambdaNorm,
			BatchSize:   batch,
			DatasetSize: data,
			TotalEpochs: T2,
		},
		Schedule: NewFixedSchedule(eta, 0),
	})
	if err != nil {
		t.Fatalf("New opt2 error: %v", err)
	}

	// Calculate expected λ values using B.1 formula: λ = λ_norm * sqrt(b / (B * T))
	lambda1 := lambdaNorm * math.Sqrt(float64(batch)/(float64(data)*float64(T1)))
	lambda2 := lambdaNorm * math.Sqrt(float64(batch)/(float64(data)*float64(T2)))

	// Verify that λ₂/λ₁ = 1/√2 when T₂ = 2*T₁
	expectedRatio := 1.0 / math.Sqrt(2.0)
	actualRatio := lambda2 / lambda1

	if math.Abs(actualRatio-expectedRatio) > 1e-12 {
		t.Fatalf("B.1 scaling verification failed:\nλ₁: %.12f\nλ₂: %.12f\nratio: %.12f\nexpected ratio: %.12f",
			lambda1, lambda2, actualRatio, expectedRatio)
	}

	// Verify that the optimizers actually use these computed λ values
	// by checking internal currentLambda() results
	computedLambda1 := opt1.currentLambda()
	computedLambda2 := opt2.currentLambda()

	if math.Abs(computedLambda1-lambda1) > 1e-12 {
		t.Fatalf("Optimizer 1 lambda mismatch: got %.12f, expected %.12f", computedLambda1, lambda1)
	}

	if math.Abs(computedLambda2-lambda2) > 1e-12 {
		t.Fatalf("Optimizer 2 lambda mismatch: got %.12f, expected %.12f", computedLambda2, lambda2)
	}
}

// TestWeightDecayB1_WarmRestarts verifies B.1 normalization with warm restarts:
// T should be taken as the length of the current restart period
func TestWeightDecayB1_WarmRestarts(t *testing.T) {
	t.Parallel()

	// Test parameters for warm restart normalization

	alpha := 0.001
	lambdaNorm := 0.1
	batch := 32
	data := 1000
	stepsPerEpoch := 10
	initialPeriod := 20 // steps = 2 epochs
	tMult := 2.0

	schedule, err := NewCosineAnnealingWarmRestarts(initialPeriod, tMult)
	if err != nil {
		t.Fatalf("Schedule creation error: %v", err)
	}

	opt, err := New([]float64{1.0}, Options{
		Alpha: alpha,
		Beta1: 0.9, Beta2: 0.999, Eps: 1e-8,
		Norm: &NormConfig{
			LambdaNorm:    lambdaNorm,
			BatchSize:     batch,
			DatasetSize:   data,
			StepsPerEpoch: stepsPerEpoch,
			TotalEpochs:   0, // not used with restarts
		},
		Schedule: schedule,
	})
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// At the beginning of first period, T should be initialPeriod/stepsPerEpoch = 2 epochs
	periodSteps, hasRestarts := opt.Schedule.PeriodInfo()
	if !hasRestarts {
		t.Fatalf("Expected warm restarts, got hasRestarts=false")
	}

	expectedPeriodSteps := initialPeriod
	if periodSteps != expectedPeriodSteps {
		t.Fatalf("Initial period mismatch: got %d steps, expected %d", periodSteps, expectedPeriodSteps)
	}

	// Verify lambda calculation uses current period length
	Ti := float64(periodSteps) / float64(stepsPerEpoch) // 20/10 = 2 epochs
	expectedLambda := lambdaNorm * math.Sqrt(float64(batch)/(float64(data)*Ti))
	actualLambda := opt.currentLambda()

	if math.Abs(actualLambda-expectedLambda) > 1e-12 {
		t.Fatalf("Warm restart lambda mismatch:\ngot:      %.12f\nexpected: %.12f\nperiod:   %d steps (%.1f epochs)",
			actualLambda, expectedLambda, periodSteps, Ti)
	}
}

// TestWeightDecayCorrectness_AllStrategies verifies that all optimization strategies
// (PureBLAS, Fusion, HeavyFusion) produce identical results with alpha scaling
func TestWeightDecayCorrectness_AllStrategies(t *testing.T) {
	t.Parallel()

	params0 := []float64{1.5, -2.5, 0.8} // base parameter values

	alpha := 0.005
	lambda := 0.05
	eta := 1.0

	// Create optimizers for different vector sizes to trigger different strategies
	testCases := []struct {
		name       string
		vectorSize int
		strategy   OptimizationStrategy
	}{
		{"Small/PureBLAS", 100, StrategyPureBLAS},
		{"Medium/Fusion", 2000, StrategyFusion},
		{"Large/HeavyFusion", 8000, StrategyHeavyFusion},
	}

	var results [][]float64

	for _, tc := range testCases {
		// Create parameter vector of the specified size, filled with test values repeated
		params := make([]float64, tc.vectorSize)
		gradients := make([]float64, tc.vectorSize)
		for i := 0; i < tc.vectorSize; i++ {
			params[i] = params0[i%len(params0)]
			gradients[i] = 0.0 // zero gradient
		}

		opt, err := New(params, Options{
			Alpha:       alpha,
			Beta1:       0.9,
			Beta2:       0.999,
			Eps:         1e-8,
			WeightDecay: lambda,
			Schedule:    NewFixedSchedule(eta, 0),
		})
		if err != nil {
			t.Fatalf("New error for %s: %v", tc.name, err)
		}

		// Verify the strategy was selected correctly
		if opt.strategy != tc.strategy {
			t.Fatalf("Strategy mismatch for %s: got %v, expected %v",
				tc.name, opt.strategy, tc.strategy)
		}

		// Perform one step
		if err := opt.Step(params, gradients); err != nil {
			t.Fatalf("Step error for %s: %v", tc.name, err)
		}

		// Store results for first few parameters for comparison
		result := make([]float64, len(params0))
		copy(result, params[:len(params0)])
		results = append(results, result)
	}

	// All strategies should produce identical results
	tolerance := 1e-12
	for i := 1; i < len(results); i++ {
		for j := 0; j < len(params0); j++ {
			if math.Abs(results[i][j]-results[0][j]) > tolerance {
				t.Fatalf("Strategy mismatch at param %d:\n%s: %.15f\n%s: %.15f\ndiff: %.2e",
					j, testCases[0].name, results[0][j],
					testCases[i].name, results[i][j],
					math.Abs(results[i][j]-results[0][j]))
			}
		}
	}

	// Verify the decay formula: θ₁ = θ₀ * (1 - α*η*λ)
	expectedFactor := 1.0 - alpha*eta*lambda
	for j := 0; j < len(params0); j++ {
		expected := params0[j] * expectedFactor
		if math.Abs(results[0][j]-expected) > tolerance {
			t.Fatalf("Alpha scaling formula verification failed at param %d:\ngot:      %.15f\nexpected: %.15f\nfactor:   %.15f",
				j, results[0][j], expected, expectedFactor)
		}
	}
}
