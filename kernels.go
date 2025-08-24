package adamw

import (
	"errors"
	"math"
)

// elementWiseSquare computes x[i] = x[i]^2 for all i
func elementWiseSquare(x []float64) {
	for i := range x {
		x[i] *= x[i]
	}
}

// clampSqrtAddEps computes x[i] = sqrt(max(x[i], 0)) + eps for all i
func clampSqrtAddEps(x []float64, eps float64) error {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
		x[i] = math.Sqrt(x[i]) + eps
		if !(x[i] > 0 && isFinite(x[i])) {
			return errors.New("invalid adaptive denominator (sqrt(vhat)+eps)")
		}
	}
	return nil
}

// elementWiseDivide computes dst[i] = num[i] / den[i] for all i
func elementWiseDivide(dst, num, den []float64) {
	for i := range dst {
		dst[i] = num[i] / den[i]
	}
}

// momentUpdateFusion performs fused m and v moment updates in one pass:
// m[i] = beta1*m[i] + (1-beta1)*g[i]
// v[i] = beta2*v[i] + (1-beta2)*g[i]^2
// Reduces memory traffic by processing both moments simultaneously
func momentUpdateFusion(m, v, g []float64, beta1, beta2 float64) {
	oneMinusBeta1 := 1.0 - beta1
	oneMinusBeta2 := 1.0 - beta2

	for i := range m {
		gi := g[i]
		m[i] = beta1*m[i] + oneMinusBeta1*gi
		v[i] = beta2*v[i] + oneMinusBeta2*gi*gi
	}
}

// biasCorrectClampSqrtFusion performs fused bias correction + clamp + sqrt in one pass:
// result[i] = sqrt(max(src[i] / bc, 0)) + eps
// Significantly reduces memory bandwidth requirements
func biasCorrectClampSqrtFusion(result, src []float64, bc, eps float64) error {
	invBC := 1.0 / bc

	for i := range result {
		val := src[i] * invBC
		if val < 0 {
			val = 0
		}
		result[i] = math.Sqrt(val) + eps
		if !(result[i] > 0 && isFinite(result[i])) {
			return errors.New("invalid adaptive denominator (sqrt(vhat)+eps)")
		}
	}
	return nil
}

// stepPureBLAS implements StrategyPureBLAS using minimal fusion
func stepPureBLAS(o *Optimizer, params, grad []float64, eta, lambda, bc1, bc2 float64) error {
	// Step 1: Simple moment updates using BLAS
	scaleVector(o.Beta1, o.m)
	axpyVector(1.0-o.Beta1, grad, o.m)

	copyVector(grad, o.vhat)
	elementWiseSquare(o.vhat)
	scaleVector(o.Beta2, o.v)
	axpyVector(1.0-o.Beta2, o.vhat, o.v)

	// Step 2: Simple bias correction
	copyVector(o.m, o.mhat)
	copyVector(o.v, o.vhat)
	scaleVector(1.0/bc1, o.mhat)
	scaleVector(1.0/bc2, o.vhat)

	// Step 3: Simple sqrt with clamp
	if err := clampSqrtAddEps(o.vhat, o.Eps); err != nil {
		return err
	}

	// Step 4: Element-wise operations
	scaleVector(o.Alpha, o.mhat)
	elementWiseDivide(o.update, o.mhat, o.vhat)

	// Step 5: Parameter updates
	if lambda > 0 {
		if o.DecayMask == nil {
			etaLambda := eta * lambda
			scaleVector(1.0-etaLambda, params)
			axpyVector(-eta, o.update, params)
		} else {
			etaLambda := eta * lambda
			for i := range o.decayUpdate {
				if o.DecayMask[i] {
					o.decayUpdate[i] = etaLambda * params[i]
				} else {
					o.decayUpdate[i] = 0
				}
			}
			subVector(o.decayUpdate, params)
			axpyVector(-eta, o.update, params)
		}
	} else {
		axpyVector(-eta, o.update, params)
	}

	return nil
}

// stepFusion implements StrategyFusion using moderate kernel fusion
func stepFusion(o *Optimizer, params, grad []float64, eta, lambda, bc1, bc2 float64) error {
	// Step 1: Fused moment updates
	momentUpdateFusion(o.m, o.v, grad, o.Beta1, o.Beta2)

	// Step 2: Bias correction for m (separate, since it doesn't need sqrt)
	copyVector(o.m, o.mhat)
	scaleVector(1.0/bc1, o.mhat)

	// Step 3: Fused bias correction + clamp + sqrt for v
	if err := biasCorrectClampSqrtFusion(o.vhat, o.v, bc2, o.Eps); err != nil {
		return err
	}

	// Step 4: Adaptive update computation
	scaleVector(o.Alpha, o.mhat)
	elementWiseDivide(o.update, o.mhat, o.vhat)

	// Step 5: Parameter updates (same as pure BLAS)
	if lambda > 0 {
		if o.DecayMask == nil {
			etaLambda := eta * lambda
			scaleVector(1.0-etaLambda, params)
			axpyVector(-eta, o.update, params)
		} else {
			etaLambda := eta * lambda
			for i := range o.decayUpdate {
				if o.DecayMask[i] {
					o.decayUpdate[i] = etaLambda * params[i]
				} else {
					o.decayUpdate[i] = 0
				}
			}
			subVector(o.decayUpdate, params)
			axpyVector(-eta, o.update, params)
		}
	} else {
		axpyVector(-eta, o.update, params)
	}

	return nil
}

// stepHeavyFusion implements StrategyHeavyFusion with maximum optimization
func stepHeavyFusion(o *Optimizer, params, grad []float64, eta, lambda, bc1, bc2 float64) error {
	// Same as fusion strategy for now
	// Future: implement more aggressive fusion kernels
	// e.g., fuse adaptive update + parameter update in one kernel
	return stepFusion(o, params, grad, eta, lambda, bc1, bc2)
}
