package main

import (
	"fmt"
	"time"

	"github.com/n0madic/go-adamw"
)

func main() {
	fmt.Println("=== AdamW Optimizer Examples ===")

	// Example 1: Traditional flat parameter optimization
	flatExample()

	fmt.Println()

	// Example 2: Matrix-based optimization (zero-copy)
	matrixExample()

	fmt.Println()

	// Example 3: Performance comparison
	performanceComparison()
}

func flatExample() {
	fmt.Println("1. Traditional Flat Parameter Optimization")
	fmt.Println("------------------------------------------")

	// Toy problem: minimize ||θ||^2 (gradient = 2θ), just to see the steps
	params := []float64{0.5, -1.0, 2.0}
	// DecayMask: apply weight decay to 1st and 3rd params, skip 2nd param (common: weights vs bias/LayerNorm)
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

	fmt.Printf("Initial params: %v\n", params)

	for step := 0; step < 100; step++ {
		grad := make([]float64, len(params))
		for i := range params {
			// gradient of quadratic: d/dθ (θ^2) = 2θ
			grad[i] = 2.0 * params[i]
		}
		if err := opt.Step(params, grad); err != nil {
			panic(err)
		}
	}
	fmt.Printf("Final params (mask=%v): %v\n", mask, params)
}

func matrixExample() {
	fmt.Println("2. Matrix-Based Optimization (Zero-Copy)")
	fmt.Println("----------------------------------------")

	// Simulate neural network parameters: weights and biases
	weights := [][]float64{
		{0.1, 0.2, 0.3}, // Layer 1 weights
		{0.4, 0.5, 0.6}, // Layer 2 weights
	}
	biases := [][]float64{
		{0.0}, // Layer 1 bias
		{0.1}, // Layer 2 bias
	}

	// Parameters as matrices (no copying required!)
	paramMatrices := [][]float64{
		weights[0], weights[1], biases[0], biases[1],
	}

	// Create optimizer for matrix parameters
	// DecayMask: apply weight decay to weights (indices 0,1,2,3,4,5), not to biases (indices 6,7)
	decayMask := []bool{true, true, true, true, true, true, false, false}

	opt, err := adamw.NewFromMatrices(paramMatrices, adamw.Options{
		Alpha:       1e-3,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 1e-2,
		Schedule:    adamw.NewFixedSchedule(1.0, 0),
		DecayMask:   decayMask,
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("Initial weights[0]: %v\n", weights[0])
	fmt.Printf("Initial weights[1]: %v\n", weights[1])
	fmt.Printf("Initial biases[0]: %v\n", biases[0])
	fmt.Printf("Initial biases[1]: %v\n", biases[1])

	for step := 0; step < 100; step++ {
		// Simulate gradients (same structure as parameters)
		gradMatrices := [][]float64{
			{2.0 * weights[0][0], 2.0 * weights[0][1], 2.0 * weights[0][2]},
			{2.0 * weights[1][0], 2.0 * weights[1][1], 2.0 * weights[1][2]},
			{2.0 * biases[0][0]},
			{2.0 * biases[1][0]},
		}

		// Optimize directly on matrix data - no copying!
		if err := opt.StepMatrices(paramMatrices, gradMatrices); err != nil {
			panic(err)
		}
	}

	fmt.Printf("Final weights[0]: %v\n", weights[0])
	fmt.Printf("Final weights[1]: %v\n", weights[1])
	fmt.Printf("Final biases[0]: %v\n", biases[0])
	fmt.Printf("Final biases[1]: %v\n", biases[1])
	fmt.Printf("(Notice: biases decay less due to decay mask)\n")
}

func performanceComparison() {
	fmt.Println("3. Performance Comparison")
	fmt.Println("-------------------------")

	// Create larger matrices for performance testing
	size := 1000
	numMatrices := 10

	// Create test data
	paramMatrices := make([][]float64, numMatrices)
	gradMatrices := make([][]float64, numMatrices)
	for i := 0; i < numMatrices; i++ {
		paramMatrices[i] = make([]float64, size)
		gradMatrices[i] = make([]float64, size)
		for j := 0; j < size; j++ {
			paramMatrices[i][j] = 0.1
			gradMatrices[i][j] = 0.01
		}
	}

	// Test 1: Traditional approach (with copying)
	start := time.Now()

	// Simulate what user had to do before: extract flat parameters
	flatParams := extractFlatParams(paramMatrices)
	flatGrads := extractFlatGrads(gradMatrices)

	opt1, err := adamw.New(flatParams, adamw.Options{
		Alpha:       1e-3,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 1e-2,
		Schedule:    adamw.NewFixedSchedule(1.0, 0),
	})
	if err != nil {
		panic(err)
	}

	for step := 0; step < 10; step++ {
		// User had to extract flat data each time (copying overhead)
		flatParams = extractFlatParams(paramMatrices)
		flatGrads = extractFlatGrads(gradMatrices)

		if err := opt1.Step(flatParams, flatGrads); err != nil {
			panic(err)
		}

		// Copy results back (more copying overhead)
		copyBackToMatrices(flatParams, paramMatrices)
	}
	traditionalTime := time.Since(start)

	// Reset data for fair comparison
	for i := 0; i < numMatrices; i++ {
		for j := 0; j < size; j++ {
			paramMatrices[i][j] = 0.1
			gradMatrices[i][j] = 0.01
		}
	}

	// Test 2: New matrix approach (zero-copy)
	start = time.Now()

	opt2, err := adamw.NewFromMatrices(paramMatrices, adamw.Options{
		Alpha:       1e-3,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 1e-2,
		Schedule:    adamw.NewFixedSchedule(1.0, 0),
	})
	if err != nil {
		panic(err)
	}

	for step := 0; step < 10; step++ {
		// Direct matrix optimization - no copying!
		if err := opt2.StepMatrices(paramMatrices, gradMatrices); err != nil {
			panic(err)
		}
	}
	matrixTime := time.Since(start)

	fmt.Printf("Traditional approach (with copying): %v\n", traditionalTime)
	fmt.Printf("New matrix approach (zero-copy):     %v\n", matrixTime)
	fmt.Printf("Performance improvement: %.2fx faster\n", float64(traditionalTime)/float64(matrixTime))
}

// Helper functions that simulate what users had to do before
func extractFlatParams(paramMatrices [][]float64) []float64 {
	totalSize := 0
	for _, matrix := range paramMatrices {
		totalSize += len(matrix)
	}

	flatParams := make([]float64, totalSize)
	offset := 0

	for _, matrix := range paramMatrices {
		copy(flatParams[offset:offset+len(matrix)], matrix)
		offset += len(matrix)
	}

	return flatParams
}

func extractFlatGrads(gradMatrices [][]float64) []float64 {
	totalSize := 0
	for _, matrix := range gradMatrices {
		totalSize += len(matrix)
	}

	flatGrads := make([]float64, totalSize)
	offset := 0

	for _, matrix := range gradMatrices {
		copy(flatGrads[offset:offset+len(matrix)], matrix)
		offset += len(matrix)
	}

	return flatGrads
}

func copyBackToMatrices(flatParams []float64, paramMatrices [][]float64) {
	offset := 0
	for _, matrix := range paramMatrices {
		copy(matrix, flatParams[offset:offset+len(matrix)])
		offset += len(matrix)
	}
}
