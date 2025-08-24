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
	lr := o.Alpha * eta
	if lambda > 0 {
		if o.DecayMask == nil {
			// UNIFORM DECAY: use full lr = alpha * eta
			etaLambda := lr * lambda
			scaleVector(1.0-etaLambda, params)
			axpyVector(-eta, o.update, params) // update already scaled by alpha
		} else {
			// MASKED DECAY
			etaLambda := lr * lambda
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

	// Step 5: Parameter updates
	lr := o.Alpha * eta
	if lambda > 0 {
		if o.DecayMask == nil {
			// UNIFORM DECAY: use full lr = alpha * eta
			etaLambda := lr * lambda
			scaleVector(1.0-etaLambda, params)
			axpyVector(-eta, o.update, params) // update already scaled by alpha
		} else {
			// MASKED DECAY
			etaLambda := lr * lambda
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

// Heavy fusion kernels for maximum optimization on large vectors

// fullMomentBiasFusion performs moment updates with immediate bias correction:
// mhat[i] = (beta1*m[i] + (1-beta1)*g[i]) / bc1
// vhat[i] = (beta2*v[i] + (1-beta2)*g[i]^2) / bc2
// Reduces memory bandwidth by combining 4 operations into 1 pass
func fullMomentBiasFusion(m, v, mhat, vhat, g []float64, beta1, beta2, bc1, bc2 float64) {
	oneMinusBeta1 := 1.0 - beta1
	oneMinusBeta2 := 1.0 - beta2
	invBC1 := 1.0 / bc1
	invBC2 := 1.0 / bc2

	for i := range m {
		gi := g[i]
		gi2 := gi * gi

		// Update moments and immediately apply bias correction
		m[i] = beta1*m[i] + oneMinusBeta1*gi
		v[i] = beta2*v[i] + oneMinusBeta2*gi2

		// Bias-corrected moments ready for use
		mhat[i] = m[i] * invBC1
		vhat[i] = v[i] * invBC2
	}
}

// adaptiveUpdateCompleteFusion performs bias correction + clamp + sqrt + scale + divide in one kernel:
// result[i] = alpha * mhat[i] / (sqrt(max(vhat[i], 0)) + eps)
// Significantly reduces memory traffic by fusing 5 operations into 1 pass
func adaptiveUpdateCompleteFusion(update, mhat, vhat []float64, alpha, eps float64) error {
	for i := range update {
		// Clamp negative values (should be rare after bias correction)
		v := vhat[i]
		if v < 0 {
			v = 0
		}

		denominator := math.Sqrt(v) + eps
		if !(denominator > 0 && isFinite(denominator)) {
			return errors.New("invalid adaptive denominator in complete fusion")
		}

		// Fused adaptive update: scale numerator and divide
		update[i] = (alpha * mhat[i]) / denominator
	}
	return nil
}

// parameterUpdateFusion performs adaptive update + weight decay + parameter update in one kernel:
// if decay: params[i] = params[i]*(1-alpha*eta*lambda) - eta*update[i]  (with optional mask)
// else:     params[i] = params[i] - eta*update[i]
// Reduces memory traffic by combining parameter update operations
func parameterUpdateFusion(params, update []float64, eta, alpha, lambda float64, decayMask []bool) {
	lrDecay := eta * alpha * lambda
	if lambda > 0 {
		if decayMask == nil {
			// Uniform decay
			oneMinusDecay := 1.0 - lrDecay
			for i := range params {
				params[i] = params[i]*oneMinusDecay - eta*update[i] // update already scaled by alpha
			}
		} else {
			// Selective decay with mask
			for i := range params {
				if decayMask[i] {
					params[i] = params[i]*(1.0-lrDecay) - eta*update[i]
				} else {
					params[i] = params[i] - eta*update[i]
				}
			}
		}
	} else {
		// No decay - simple update
		for i := range params {
			params[i] = params[i] - eta*update[i]
		}
	}
}

// stepHeavyFusion implements StrategyHeavyFusion with maximum optimization
func stepHeavyFusion(o *Optimizer, params, grad []float64, eta, lambda, bc1, bc2 float64) error {
	// Heavy fusion approach: minimize memory passes for large vectors

	// Step 1: Fused moment updates with immediate bias correction
	// This replaces both momentUpdateFusion + separate bias correction
	fullMomentBiasFusion(o.m, o.v, o.mhat, o.vhat, grad, o.Beta1, o.Beta2, bc1, bc2)

	// Step 2: Complete adaptive update fusion
	// This replaces: clamp + sqrt + scale + divide operations
	if err := adaptiveUpdateCompleteFusion(o.update, o.mhat, o.vhat, o.Alpha, o.Eps); err != nil {
		return err
	}

	// Step 3: Fused parameter update with decay
	// This replaces separate decay and parameter update operations
	parameterUpdateFusion(params, o.update, eta, o.Alpha, lambda, o.DecayMask)

	return nil
}
