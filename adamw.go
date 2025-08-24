// File: adamw.go
// Go 1.20+
//
// AdamW / AdamWR per Loshchilov & Hutter (ICLR 2019)
// - Decoupled weight decay (Algorithm 2, step 12)
// - Normalized weight decay (Appendix B.1)
// - Cosine annealing with warm restarts (Appendix B.2, Eq. 15)
//
// Engineering hardening:
// - Strict hyperparameter validation (beta bounds, eps>0, alpha>0)
// - Param length vs. internal state length check
// - Non-finite gradient/param guards
// - Soft clamp of vhat (vhat = max(vhat, 0)) before sqrt
// - Optional weight-decay mask to exclude certain parameters from decay
//
// Concurrency: this optimizer is NOT goroutine-safe. If multiple goroutines
// update the same Optimizer or share the same parameter slice, you must
// synchronize externally.

package adamw

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/blas/blas64"
)

// Optimizer implements AdamW/AdamWR for an in-place updated parameter vector θ.
type Optimizer struct {
	// Adam hyperparameters
	Alpha float64 // base lr (α), must be > 0
	Beta1 float64 // β1 in [0,1)
	Beta2 float64 // β2 in [0,1)
	Eps   float64 // ε > 0

	// Weight decay:
	// If Norm != nil, λ is derived from λ_norm via B.1 (see currentLambda).
	// Otherwise WeightDecay (λ) is used directly (λ<0 interpreted as no decay).
	WeightDecay float64
	Norm        *NormConfig

	// Optional mask to disable decay for certain parameters (e.g., bias/LayerNorm).
	// If nil, decay applies to all parameters. If non-nil, its length must equal len(params).
	DecayMask []bool

	// Schedule multiplier η_t
	Schedule Schedule

	// Internal state
	t        int64
	powBeta1 float64
	powBeta2 float64
	m, v     []float64

	// Working buffers to avoid allocations
	mhat, vhat, update, decayUpdate []float64

	// Adaptive optimization configuration
	adaptiveConfig AdaptiveConfig
	strategy       OptimizationStrategy

	initCalled bool
}

// NormConfig — B.1 normalization: λ = λ_norm * sqrt(b / (B * T))
//
//	b = BatchSize (>0)
//	B = DatasetSize (>0)
//	T = TotalEpochs (>0) for fixed schedules, or length of current restart (epochs) for WR.
type NormConfig struct {
	LambdaNorm    float64 // λ_norm (>=0 recommended)
	BatchSize     int     // b (>0)
	DatasetSize   int     // B (>0)
	TotalEpochs   int     // T (>=0; used if no restarts)
	StepsPerEpoch int     // (>0 recommended if using warm restarts; used to convert period steps -> epochs)
}

// Schedule defines η_t and allows ticking per update.
type Schedule interface {
	// Eta returns η_t for the CURRENT step (before Tick()).
	Eta() float64
	// Tick advances the internal step by 1.
	Tick()
	// Reset resets the schedule state.
	Reset()
	// PeriodInfo returns (periodSteps, hasRestarts).
	PeriodInfo() (periodSteps int, hasRestarts bool)
}

// ---------- Schedules ----------

// FixedSchedule — constant η_t (defaults to 1 if <=0).
type FixedSchedule struct {
	eta        float64
	totalSteps int
}

func NewFixedSchedule(eta float64, totalSteps int) *FixedSchedule {
	if eta <= 0 {
		eta = 1.0
	}
	return &FixedSchedule{eta: eta, totalSteps: totalSteps}
}
func (s *FixedSchedule) Eta() float64            { return s.eta }
func (s *FixedSchedule) Tick()                   {}
func (s *FixedSchedule) Reset()                  {}
func (s *FixedSchedule) PeriodInfo() (int, bool) { return s.totalSteps, false }

// CosineAnnealingWarmRestarts — Eq. (15), periods in steps, period grows by tMult on restart.
type CosineAnnealingWarmRestarts struct {
	initialPeriodSteps int
	tMult              float64
	curPeriodSteps     int
	tcur               int
}

func NewCosineAnnealingWarmRestarts(initialPeriodSteps int, tMult float64) (*CosineAnnealingWarmRestarts, error) {
	if initialPeriodSteps <= 0 {
		return nil, errors.New("initialPeriodSteps must be > 0")
	}
	if tMult < 1.0 {
		return nil, errors.New("tMult must be >= 1.0")
	}
	return &CosineAnnealingWarmRestarts{
		initialPeriodSteps: initialPeriodSteps,
		tMult:              tMult,
		curPeriodSteps:     initialPeriodSteps,
		tcur:               0,
	}, nil
}

func (s *CosineAnnealingWarmRestarts) Eta() float64 {
	// η_t = 0.5 + 0.5 * cos(pi * Tcur / Ti), Tcur ∈ [0, Ti].
	r := float64(s.tcur) / float64(s.curPeriodSteps)
	return 0.5 + 0.5*math.Cos(math.Pi*r)
}

func (s *CosineAnnealingWarmRestarts) Tick() {
	s.tcur++
	if s.tcur >= s.curPeriodSteps {
		// restart
		s.tcur = 0
		s.curPeriodSteps = int(math.Round(float64(s.curPeriodSteps) * s.tMult))
		if s.curPeriodSteps <= 0 {
			s.curPeriodSteps = 1
		}
	}
}

func (s *CosineAnnealingWarmRestarts) Reset() {
	s.curPeriodSteps = s.initialPeriodSteps
	s.tcur = 0
}

func (s *CosineAnnealingWarmRestarts) PeriodInfo() (int, bool) {
	return s.curPeriodSteps, true
}

// ---------- Constructor & validation ----------

type Options struct {
	Alpha       float64
	Beta1       float64
	Beta2       float64
	Eps         float64
	WeightDecay float64
	Norm        *NormConfig
	Schedule    Schedule
	DecayMask   []bool // optional; if provided, must match params length
}

func New(params []float64, opt Options) (*Optimizer, error) {
	if len(params) == 0 {
		return nil, errors.New("params must be non-empty")
	}

	// Configure adaptive optimization based on vector size
	adaptiveConfig := DefaultAdaptiveConfig()
	strategy := SelectOptimizationStrategy(len(params), adaptiveConfig)

	o := &Optimizer{
		Alpha:          ifPositiveOr(opt.Alpha, 1e-3),
		Beta1:          opt.Beta1,
		Beta2:          opt.Beta2,
		Eps:            ifPositiveOr(opt.Eps, 1e-8),
		WeightDecay:    opt.WeightDecay,
		Norm:           opt.Norm,
		Schedule:       opt.Schedule,
		DecayMask:      nil,
		adaptiveConfig: adaptiveConfig,
		strategy:       strategy,
		t:              0,
		powBeta1:       1.0,
		powBeta2:       1.0,
		m:              make([]float64, len(params)),
		v:              make([]float64, len(params)),

		// Pre-allocate working buffers
		mhat:        make([]float64, len(params)),
		vhat:        make([]float64, len(params)),
		update:      make([]float64, len(params)),
		decayUpdate: make([]float64, len(params)),
	}

	// Defaults for betas if not provided (<=0): conventional values.
	if o.Beta1 <= 0 {
		o.Beta1 = 0.9
	}
	if o.Beta2 <= 0 {
		o.Beta2 = 0.999
	}
	// Strict validation per engineering hardening.
	if !(o.Beta1 >= 0.0 && o.Beta1 < 1.0) {
		return nil, errors.New("beta1 must be in [0,1)")
	}
	if !(o.Beta2 >= 0.0 && o.Beta2 < 1.0) {
		return nil, errors.New("beta2 must be in [0,1)")
	}
	if !(o.Alpha > 0.0) {
		return nil, errors.New("alpha must be > 0")
	}
	if !(o.Eps > 0.0) {
		return nil, errors.New("eps must be > 0")
	}

	if o.Schedule == nil {
		o.Schedule = NewFixedSchedule(1.0, 0)
	}

	// Validate normalization config, if present.
	if o.Norm != nil {
		if o.Norm.LambdaNorm < 0 {
			return nil, errors.New("Norm.LambdaNorm must be >= 0")
		}
		if o.Norm.BatchSize <= 0 || o.Norm.DatasetSize <= 0 {
			return nil, errors.New("Norm.BatchSize and Norm.DatasetSize must be > 0")
		}
		if o.Norm.TotalEpochs < 0 {
			return nil, errors.New("Norm.TotalEpochs must be >= 0")
		}
		if o.Norm.StepsPerEpoch < 0 {
			return nil, errors.New("Norm.StepsPerEpoch must be >= 0")
		}
	}

	// Copy DecayMask if provided and validate length.
	if opt.DecayMask != nil {
		if len(opt.DecayMask) != len(params) {
			return nil, errors.New("DecayMask length must match params length")
		}
		o.DecayMask = make([]bool, len(opt.DecayMask))
		copy(o.DecayMask, opt.DecayMask)
	}

	o.initCalled = true
	return o, nil
}

func ifPositiveOr(v, def float64) float64 {
	if v > 0 {
		return v
	}
	return def
}

// ---------- Utility ----------

func isFinite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// toVector creates a blas64.Vector from a float64 slice for BLAS operations
func toVector(data []float64) blas64.Vector {
	return blas64.Vector{N: len(data), Data: data, Inc: 1}
}

// BLAS-optimized vector operations
func scaleVector(alpha float64, x []float64) {
	blas64.Scal(alpha, toVector(x))
}

func axpyVector(alpha float64, x, y []float64) {
	// y = alpha*x + y
	blas64.Axpy(alpha, toVector(x), toVector(y))
}

func copyVector(x, y []float64) {
	blas64.Copy(toVector(x), toVector(y))
}

func subVector(x, y []float64) {
	// y = y - x  (equivalent to floats.Sub(y, x))
	axpyVector(-1.0, x, y)
}

// ---------- Main update ----------

// Step performs one AdamW update on `params` using `grad`.
// Requirements:
// - len(params) == len(grad) == len(internal state)
// - all inputs finite
func (o *Optimizer) Step(params, grad []float64) error {
	if !o.initCalled {
		return errors.New("optimizer not initialized")
	}
	if len(params) == 0 || len(grad) == 0 {
		return errors.New("params and grad must be non-empty")
	}
	if len(params) != len(grad) {
		return errors.New("params and grad must have equal length")
	}
	if len(params) != len(o.m) || len(params) != len(o.v) {
		return errors.New("params length must match optimizer state length (m/v)")
	}
	if o.DecayMask != nil && len(o.DecayMask) != len(params) {
		return errors.New("DecayMask length must match params length")
	}
	// Finite guards
	for i := 0; i < len(params); i++ {
		if !isFinite(params[i]) {
			return errors.New("non-finite parameter encountered")
		}
		if !isFinite(grad[i]) {
			return errors.New("non-finite gradient encountered")
		}
	}

	// t := t + 1
	o.t++

	// Update powers for bias correction factors: (1 - β^t)
	o.powBeta1 *= o.Beta1
	o.powBeta2 *= o.Beta2
	bc1 := 1.0 - o.powBeta1
	bc2 := 1.0 - o.powBeta2
	if !(bc1 > 0.0 && bc2 > 0.0 && isFinite(bc1) && isFinite(bc2)) {
		return errors.New("invalid bias-correction denominators")
	}

	eta := o.Schedule.Eta()
	// Note: if eta<=0, neither gradient nor decay is applied (consistent with cosine minima).
	if eta < 0 {
		eta = 0
	}

	lambda := o.currentLambda()

	// Adaptive strategy selection: use pre-computed strategy
	var err error
	switch o.strategy {
	case StrategyPureBLAS:
		err = stepPureBLAS(o, params, grad, eta, lambda, bc1, bc2)
	case StrategyFusion:
		err = stepFusion(o, params, grad, eta, lambda, bc1, bc2)
	case StrategyHeavyFusion:
		err = stepHeavyFusion(o, params, grad, eta, lambda, bc1, bc2)
	default:
		// Fallback to fusion strategy
		err = stepFusion(o, params, grad, eta, lambda, bc1, bc2)
	}

	if err != nil {
		return err
	}

	o.Schedule.Tick()
	return nil
}

// CurrentStep returns t (starting from 1 after first Step()).
func (o *Optimizer) CurrentStep() int64 { return o.t }

// ResetState clears moments and counters (keeps hyperparameters).
func (o *Optimizer) ResetState() {
	for i := range o.m {
		o.m[i] = 0
		o.v[i] = 0
		o.mhat[i] = 0
		o.vhat[i] = 0
		o.update[i] = 0
		o.decayUpdate[i] = 0
	}
	o.t = 0
	o.powBeta1 = 1.0
	o.powBeta2 = 1.0
	if o.Schedule != nil {
		o.Schedule.Reset()
	}
}

// currentLambda computes λ: fixed or normalized (B.1).
func (o *Optimizer) currentLambda() float64 {
	if o.Norm == nil {
		if o.WeightDecay < 0 {
			return 0
		}
		return o.WeightDecay
	}
	// λ = λ_norm * sqrt(b / (B * T))
	b := float64(o.Norm.BatchSize)
	B := float64(o.Norm.DatasetSize)

	// T: inferred either from restarts or from TotalEpochs
	T := float64(o.Norm.TotalEpochs)
	if stepsPerEpoch := o.Norm.StepsPerEpoch; stepsPerEpoch > 0 && o.Schedule != nil {
		if periodSteps, hasR := o.Schedule.PeriodInfo(); hasR && periodSteps > 0 {
			T = float64(periodSteps) / float64(stepsPerEpoch)
			if T < 1.0 {
				T = 1.0
			}
		}
	}
	if T <= 0 {
		// Fallback: if no restarts info and TotalEpochs not set, use T=1
		T = 1.0
	}
	l := o.Norm.LambdaNorm * math.Sqrt(b/(B*T))
	if l < 0 {
		return 0
	}
	return l
}
