package adamw

import (
	"math"
	"testing"
)

// clamp helpers
func clamp(x, lo, hi float64) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return lo
	}
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

func clampBeta(x float64) float64 {
	// keep strictly below 1 to avoid division by zero in bias-correction
	return clamp(x, 0.0, 1.0-1e-12)
}

func allFinite(xs []float64) bool {
	for _, v := range xs {
		if !isFinite(v) {
			return false
		}
	}
	return true
}

func buildGradient(dim int, mag float64) []float64 {
	g := make([]float64, dim)
	for i := 0; i < dim; i++ {
		// Deterministic pattern with extremes:
		// mixture of zeros, tiny, big, and alternating signs
		val := mag * math.Sin(float64(i)*1.731+0.123) // base
		if i%7 == 0 {
			val = mag * 1e-12
		}
		if i%11 == 0 {
			val = -mag
		}
		if i%13 == 0 {
			val = 0.0
		}
		g[i] = val
	}
	return g
}

func buildParams(dim int) []float64 {
	p := make([]float64, dim)
	for i := 0; i < dim; i++ {
		// Small but non-trivial initialization (deterministic)
		p[i] = 1e-2 * math.Cos(float64(i)*0.777+0.456)
	}
	return p
}

// chooseSchedule builds either FixedSchedule or CosineAnnealingWarmRestarts
// deterministically from provided knobs.
func chooseSchedule(useCosine bool, eta float64, stepsHint int) Schedule {
	if !useCosine {
		// Fixed eta in [1e-6, 10]
		return NewFixedSchedule(clamp(eta, 1e-6, 10.0), 0)
	}
	// Cosine with warm restarts: period in steps within [2, 256], tMult in [1, 3)
	period := stepsHint
	if period < 2 {
		period = 2
	}
	if period > 256 {
		period = 256
	}
	tmult := 1.0 + math.Mod(eta, 2.0) // in [1,3)
	s, err := NewCosineAnnealingWarmRestarts(period, tmult)
	if err != nil {
		// Fallback should not happen with the clamps above, but be safe.
		return NewFixedSchedule(1.0, 0)
	}
	return s
}

// buildOptions deterministically picks either fixed lambda or normalized lambda.
// IMPORTANT: we bound lambda to avoid explosive |1 - eta*lambda| > 1 growth.
func buildOptions(alpha, b1, b2, eps, lambda, eta, gradMag float64, dim, steps int, useCosine bool) Options {
	// Safer numeric ranges to avoid trivial overflow
	alpha = clamp(alpha, 1e-8, 1.0)
	b1 = clampBeta(b1)
	b2 = clamp(b2, 0.90, 1.0-1e-12)
	eps = clamp(eps, 1e-12, 1e-2)

	// Schedule + etaMax bound (maximal multiplier value)
	sched := chooseSchedule(useCosine, eta, steps)
	etaMax := 1.0
	if !useCosine {
		etaMax = clamp(eta, 1e-6, 10.0)
	}
	// To keep per-step decay contractive, enforce eta*lambda <= 0.9
	lamMax := 0.9 / etaMax
	lambda = clamp(lambda, 0.0, lamMax)

	// 50/50: use normalized lambda or fixed lambda depending on gradMag
	useNorm := math.Mod(gradMag, 2.0) >= 1.0

	if !useNorm {
		return Options{
			Alpha:       alpha,
			Beta1:       b1,
			Beta2:       b2,
			Eps:         eps,
			WeightDecay: lambda,
			Schedule:    sched,
		}
	}

	// Normalized lambda: pick plausible dataset/batch/spe to compute T or T_i.
	batch := int(clamp(math.Round(1.0+math.Mod(float64(dim)*3.0, 256.0)), 1.0, 4096.0))
	data := int(clamp(float64(10*batch), 100.0, 10_000_000.0))
	stepsPerEpoch := int(clamp(math.Round(math.Mod(float64(dim), 64.0)+4.0), 1.0, 10_000.0))

	// If cosine schedule is used, TotalEpochs is ignored;
	// otherwise we need a positive T for normalization.
	totalEpochs := int(clamp(math.Round(math.Mod(float64(steps), 50.0)+1.0), 1.0, 10000.0))

	// Bound lambda_norm so that worst-case λ (with T_min=1) satisfies eta*λ <= 0.9
	ratio := math.Sqrt(float64(batch) / (float64(data) * 1.0)) // T_min=1
	lamNormMax := lamMax
	if ratio > 0 {
		lamNormMax = lamMax / ratio
	}
	lamNorm := clamp(lambda, 0.0, lamNormMax) // reuse "lambda" input slot as λ_norm seed

	return Options{
		Alpha: alpha, Beta1: b1, Beta2: b2, Eps: eps,
		Norm: &NormConfig{
			LambdaNorm:    lamNorm,
			BatchSize:     batch,
			DatasetSize:   data,
			TotalEpochs:   totalEpochs,
			StepsPerEpoch: stepsPerEpoch,
		},
		Schedule: sched,
	}
}

// FuzzStepStability stresses the optimizer with extreme hyperparameters and gradients,
// asserting numerical stability invariants (no NaN/Inf, non-negative v_t, valid bias-correction).
func FuzzStepStability(f *testing.F) {
	// Seed cases (covering zeros, tiny/huge grads, near-1 betas, tiny eps)
	f.Add(8, 50, 1e-3, 0.9, 0.999, 1e-12, 1e-2, 1.0, 1.0)
	f.Add(32, 80, 1e-6, 0.0, 0.999999, 1e-12, 0.0, 1.0, 1e6)
	f.Add(64, 40, 1.0, 0.999999, 0.999999999, 1e-12, 10.0, 0.3, 1e-12)
	f.Add(4, 10, 10.0, 0.5, 0.95, 1e-6, 1e2, 0.9, 1e3)
	f.Add(16, 30, 5e-2, 1.0-1e-14, 0.9999, 1e-9, 1e-2, 1e-6, 0.0)
	f.Add(3, 5, 1e-4, 0.1, 0.999, 1e-4, 0.0, 2.0, 1e2)

	f.Fuzz(func(t *testing.T,
		dimIn, stepsIn int,
		alphaIn, b1In, b2In, epsIn, lambdaIn, etaIn, gradMagIn float64,
	) {

		// Sanitize dimensions and steps
		dim := int(clamp(float64(dimIn), 1.0, 512.0))
		steps := int(clamp(float64(stepsIn), 1.0, 500.0))

		// Build gradients and params
		gradMag := clamp(gradMagIn, 0.0, 1e12)
		grad := buildGradient(dim, gradMag)
		params := buildParams(dim)

		// Deterministically choose schedule kind from inputs
		useCosine := (math.Mod(alphaIn+etaIn, 2.0) >= 1.0)

		opts := buildOptions(alphaIn, b1In, b2In, epsIn, lambdaIn, etaIn, gradMag, dim, steps, useCosine)

		// Construct optimizer
		opt, err := New(clone(params), opts)
		if err != nil {
			// Construction should succeed with our clamps; if not, it's a bug.
			t.Fatalf("New() error: %v (opts=%+v)", err, opts)
		}

		// Run multiple steps and assert invariants
		for s := 0; s < steps; s++ {
			if err := opt.Step(params, grad); err != nil {
				t.Fatalf("Step error at %d: %v", s, err)
			}
			// Invariants on params
			if !allFinite(params) {
				t.Fatalf("non-finite params at step %d: %#v", s, params)
			}
			// Invariants on moments m, v
			if !allFinite(opt.m) || !allFinite(opt.v) {
				t.Fatalf("non-finite moments at step %d", s)
			}
			// v_t must be >= 0 within numerical noise
			for i := range opt.v {
				if opt.v[i] < -1e-18 {
					t.Fatalf("v[%d]=%g < 0 at step %d", i, opt.v[i], s)
				}
			}
			// Bias-correction denominators must be positive and finite
			bc1 := 1.0 - opt.powBeta1
			bc2 := 1.0 - opt.powBeta2
			if !(isFinite(bc1) && isFinite(bc2) && bc1 > 0 && bc2 > 0) {
				t.Fatalf("invalid bias-correction at step %d: bc1=%g bc2=%g", s, bc1, bc2)
			}
			// sqrt(vhat)+eps > 0 and finite
			// Note: eps is opts.Eps (stored in opt.Eps); vhat = v / bc2
			for i := range opt.v {
				vhat := opt.v[i] / bc2
				den := math.Sqrt(math.Max(vhat, 0)) + opt.Eps
				if !(isFinite(den) && den > 0) {
					t.Fatalf("invalid denominator at step %d (i=%d): vhat=%g eps=%g den=%g",
						s, i, vhat, opt.Eps, den)
				}
			}
		}

		// If cosine schedule: check its range and period behavior minimally
		if _, hasR := opt.Schedule.PeriodInfo(); hasR {
			etaNow := opt.Schedule.Eta()
			if etaNow < 0 || etaNow > 1 || !isFinite(etaNow) {
				t.Fatalf("cosine Eta out of [0,1]: %g", etaNow)
			}
		}
	})
}

// FuzzOneStepMatchesManual verifies that a single AdamW update matches
// an independent manual oracle (bias-corrected Adam + decoupled weight decay).
// We keep schedule fixed (eta constant) here to compare formulas directly.
func FuzzOneStepMatchesManual(f *testing.F) {
	// Seeds covering typical and extreme corners (still numerically safe)
	f.Add(8, 1e-3, 0.9, 0.999, 1e-8, 1e-2, 1.0, 1.0)
	f.Add(32, 1e-6, 0.0, 0.999999, 1e-8, 0.0, 0.3, 1e6)
	f.Add(3, 1.0, 0.999999, 0.999999999, 1e-8, 1e-3, 10.0, 1e-2)
	f.Add(16, 5e-2, 1.0-1e-14, 0.9999, 1e-8, 1e-2, 1e-6, 0.0)
	f.Add(4, 1e-4, 0.1, 0.999, 1e-6, 0.0, 2.0, 1e2)

	f.Fuzz(func(t *testing.T,
		dimIn int,
		alphaIn, b1In, b2In, epsIn, lambdaIn, etaIn, gradMagIn float64,
	) {
		// Dimensions and numeric clamps (safe ranges)
		dim := int(clamp(float64(dimIn), 1.0, 128.0))
		alpha := clamp(alphaIn, 1e-8, 1.0)
		b1 := clampBeta(b1In)
		b2 := clamp(b2In, 0.90, 1.0-1e-12)
		eps := clamp(epsIn, 1e-12, 1e-2)
		eta := clamp(etaIn, 1e-6, 10.0)
		// Keep per-step decay contractive to avoid trivial explosion:
		lamMax := 0.9 / eta
		lambda := clamp(lambdaIn, 0.0, lamMax)

		// Params and gradient
		gradMag := clamp(gradMagIn, 0.0, 1e9)
		params0 := buildParams(dim)
		grad := buildGradient(dim, gradMag)

		// Library step (fixed schedule = eta)
		opt, err := New(clone(params0), Options{
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
		lib := clone(params0)
		if err := opt.Step(lib, grad); err != nil {
			t.Fatalf("Step error: %v", err)
		}

		// Manual oracle: one step AdamW with bias-correction + decoupled decay
		man := clone(params0)
		ms := newManualState(dim)
		emulateOneStepManual(man, grad, ms, alpha, b1, b2, eps, eta, lambda)

		// Compare with tight tolerances
		absTol, relTol := 1e-12, 1e-10
		if !slicesAlmostEqual(lib, man, absTol, relTol) {
			// Provide additional context for debugging
			var maxRel float64
			for i := range lib {
				a, b := lib[i], man[i]
				num := math.Abs(a - b)
				den := math.Max(math.Abs(a), math.Abs(b))
				if den > 0 {
					if r := num / den; r > maxRel {
						maxRel = r
					}
				}
			}
			t.Fatalf("one-step mismatch:\nlib=%#v\nman=%#v\nmaxRel=%.3e (α=%g β1=%g β2=%g ε=%g η=%g λ=%g)",
				lib, man, maxRel, alpha, b1, b2, eps, eta, lambda)
		}
	})
}
