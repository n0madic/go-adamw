package main

import (
	"fmt"

	"github.com/n0madic/go-adamw"
)

func main() {
	// Toy problem: minimize ||θ||^2 (gradient = 2θ), just to see the steps
	params := []float64{0.5, -1.0, 2.0}
	// DecayMask: disable weight decay for the second parameter (common case: bias/LayerNorm)
	mask := []bool{true, false, true}

	opt, err := adamw.New(params, adamw.Options{
		Alpha:       1e-3,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 1e-2,                           // decoupled λ
		Schedule:    adamw.NewFixedSchedule(1.0, 0), // η_t = 1
		DecayMask:   mask,
	})
	if err != nil {
		panic(err)
	}

	for step := 0; step < 1000; step++ {
		grad := make([]float64, len(params))
		for i := range params {
			// gradient of quadratic: d/dθ (θ^2) = 2θ
			grad[i] = 2.0 * params[i]
		}
		if err := opt.Step(params, grad); err != nil {
			panic(err)
		}
	}
	fmt.Printf("params (mask=%#v): %#v\n", mask, params)
}
