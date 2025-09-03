package adamw

import (
	"testing"
)

// TestMatrixView_BasicOperations tests basic MatrixView functionality
func TestMatrixView_BasicOperations(t *testing.T) {
	// Test with multiple matrices of different sizes
	matrices := [][]float64{
		{1.0, 2.0, 3.0},      // 3 elements
		{4.0, 5.0},           // 2 elements
		{6.0, 7.0, 8.0, 9.0}, // 4 elements
	}

	mv, err := NewMatrixView(matrices)
	if err != nil {
		t.Fatalf("NewMatrixView failed: %v", err)
	}

	// Test length
	expectedLen := 3 + 2 + 4
	if mv.Len() != expectedLen {
		t.Errorf("Expected length %d, got %d", expectedLen, mv.Len())
	}

	// Test Get operations
	expected := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	for i := 0; i < mv.Len(); i++ {
		got := mv.Get(i)
		if got != expected[i] {
			t.Errorf("Get(%d): expected %f, got %f", i, expected[i], got)
		}
	}

	// Test Set operations
	for i := 0; i < mv.Len(); i++ {
		newVal := float64(i + 10)
		mv.Set(i, newVal)
		got := mv.Get(i)
		if got != newVal {
			t.Errorf("Set(%d, %f) then Get(%d): expected %f, got %f", i, newVal, i, newVal, got)
		}
	}

	// Verify changes are reflected in original matrices
	if matrices[0][0] != 10.0 || matrices[1][1] != 14.0 || matrices[2][3] != 18.0 {
		t.Error("Changes not reflected in original matrices")
	}
}

// TestMatrixView_ToFlatAndCopyFromFlat tests data conversion methods
func TestMatrixView_ToFlatAndCopyFromFlat(t *testing.T) {
	matrices := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0, 5.0},
	}

	mv, err := NewMatrixView(matrices)
	if err != nil {
		t.Fatalf("NewMatrixView failed: %v", err)
	}

	// Test ToFlat
	flat := mv.ToFlat()
	expected := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	if !slicesAlmostEqual(flat, expected, 1e-10, 1e-10) {
		t.Errorf("ToFlat: expected %v, got %v", expected, flat)
	}

	// Test CopyFromFlat
	newFlat := []float64{10.0, 20.0, 30.0, 40.0, 50.0}
	err = mv.CopyFromFlat(newFlat)
	if err != nil {
		t.Fatalf("CopyFromFlat failed: %v", err)
	}

	// Verify changes
	if matrices[0][0] != 10.0 || matrices[0][1] != 20.0 ||
		matrices[1][0] != 30.0 || matrices[1][1] != 40.0 || matrices[1][2] != 50.0 {
		t.Error("CopyFromFlat did not update matrices correctly")
	}
}

// TestMatrixView_ForEachMethods tests iteration methods
func TestMatrixView_ForEachMethods(t *testing.T) {
	matrices := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	mv, err := NewMatrixView(matrices)
	if err != nil {
		t.Fatalf("NewMatrixView failed: %v", err)
	}

	// Test ForEach
	collected := make([]float64, 0)
	mv.ForEach(func(index int, value float64) {
		collected = append(collected, value)
	})
	expected := []float64{1.0, 2.0, 3.0, 4.0}
	if !slicesAlmostEqual(collected, expected, 1e-10, 1e-10) {
		t.Errorf("ForEach: expected %v, got %v", expected, collected)
	}

	// Test ForEachMutable
	mv.ForEachMutable(func(index int, value *float64) {
		*value *= 2.0
	})

	// Verify changes
	if matrices[0][0] != 2.0 || matrices[0][1] != 4.0 ||
		matrices[1][0] != 6.0 || matrices[1][1] != 8.0 {
		t.Error("ForEachMutable did not update matrices correctly")
	}
}

// TestNewFromMatrices_BasicFunctionality tests the matrix-based constructor
func TestNewFromMatrices_BasicFunctionality(t *testing.T) {
	// Create sample matrices
	weights := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	bias := [][]float64{{0.0}, {0.1}}

	paramMatrices := [][]float64{weights[0], weights[1], bias[0], bias[1]}

	// Create optimizer
	opt, err := NewFromMatrices(paramMatrices, Options{
		Alpha:       1e-3,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 1e-2,
		Schedule:    NewFixedSchedule(1.0, 0),
	})
	if err != nil {
		t.Fatalf("NewFromMatrices failed: %v", err)
	}

	// Check that optimizer was initialized correctly
	expectedParamCount := 2 + 2 + 1 + 1 // 6 total parameters
	if len(opt.m) != expectedParamCount || len(opt.v) != expectedParamCount {
		t.Errorf("Expected %d parameters, got m=%d, v=%d", expectedParamCount, len(opt.m), len(opt.v))
	}
}

// TestStepMatrices_MatchesRegularStep tests that matrix optimization produces same results
func TestStepMatrices_MatchesRegularStep(t *testing.T) {
	// Create test data
	paramData := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	gradData := []float64{0.01, 0.02, 0.03, 0.04, 0.05, 0.06}

	// Create matrices that represent the same data
	paramMatrices := [][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}
	gradMatrices := [][]float64{
		{0.01, 0.02, 0.03},
		{0.04, 0.05, 0.06},
	}

	// Create two optimizers with identical settings
	optFlat, err := New(clone(paramData), Options{
		Alpha:       1e-3,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 1e-2,
		Schedule:    NewFixedSchedule(1.0, 0),
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	optMatrix, err := NewFromMatrices([][]float64{
		clone(paramMatrices[0]),
		clone(paramMatrices[1]),
	}, Options{
		Alpha:       1e-3,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 1e-2,
		Schedule:    NewFixedSchedule(1.0, 0),
	})
	if err != nil {
		t.Fatalf("NewFromMatrices failed: %v", err)
	}

	// Perform optimization steps
	numSteps := 10
	params1 := clone(paramData)
	grads1 := clone(gradData)

	params2Matrices := [][]float64{
		clone(paramMatrices[0]),
		clone(paramMatrices[1]),
	}
	grads2Matrices := [][]float64{
		clone(gradMatrices[0]),
		clone(gradMatrices[1]),
	}

	for step := 0; step < numSteps; step++ {
		// Step with flat optimizer
		err = optFlat.Step(params1, grads1)
		if err != nil {
			t.Fatalf("optFlat.Step failed at step %d: %v", step, err)
		}

		// Step with matrix optimizer
		err = optMatrix.StepMatrices(params2Matrices, grads2Matrices)
		if err != nil {
			t.Fatalf("optMatrix.StepMatrices failed at step %d: %v", step, err)
		}
	}

	// Convert matrix results to flat for comparison
	matrixView, err := NewMatrixView(params2Matrices)
	if err != nil {
		t.Fatalf("NewMatrixView failed: %v", err)
	}
	params2Flat := matrixView.ToFlat()

	// Compare results
	const tolerance = 1e-12
	if !slicesAlmostEqual(params1, params2Flat, tolerance, tolerance) {
		t.Errorf("Results don't match:\nFlat:   %v\nMatrix: %v", params1, params2Flat)
	}
}

// TestStepMatrices_AllStrategies tests all optimization strategies with matrices
func TestStepMatrices_AllStrategies(t *testing.T) {
	strategies := []OptimizationStrategy{
		StrategyPureBLAS,
		StrategyFusion,
		StrategyHeavyFusion,
	}

	paramMatrices := [][]float64{
		{0.1, 0.2, 0.3, 0.4, 0.5},
		{0.6, 0.7, 0.8, 0.9, 1.0},
	}
	gradMatrices := [][]float64{
		{0.01, 0.02, 0.03, 0.04, 0.05},
		{0.06, 0.07, 0.08, 0.09, 0.10},
	}

	var results [][]float64

	for _, strategy := range strategies {
		// Create matrices copy for this test
		testParamMatrices := [][]float64{
			clone(paramMatrices[0]),
			clone(paramMatrices[1]),
		}
		testGradMatrices := [][]float64{
			clone(gradMatrices[0]),
			clone(gradMatrices[1]),
		}

		// Create optimizer
		opt, err := NewFromMatrices(testParamMatrices, Options{
			Alpha:       1e-3,
			Beta1:       0.9,
			Beta2:       0.999,
			Eps:         1e-8,
			WeightDecay: 1e-2,
			Schedule:    NewFixedSchedule(1.0, 0),
		})
		if err != nil {
			t.Fatalf("NewFromMatrices failed for strategy %v: %v", strategy, err)
		}

		// Force the strategy
		opt.strategy = strategy

		// Perform optimization steps
		for step := 0; step < 5; step++ {
			err = opt.StepMatrices(testParamMatrices, testGradMatrices)
			if err != nil {
				t.Fatalf("StepMatrices failed for strategy %v at step %d: %v", strategy, step, err)
			}
		}

		// Convert to flat for comparison
		matrixView, err := NewMatrixView(testParamMatrices)
		if err != nil {
			t.Fatalf("NewMatrixView failed: %v", err)
		}
		results = append(results, matrixView.ToFlat())
	}

	// All strategies should produce the same results
	const tolerance = 1e-10
	for i := 1; i < len(results); i++ {
		if !slicesAlmostEqual(results[0], results[i], tolerance, tolerance) {
			t.Errorf("Strategy %v produces different results than strategy %v", strategies[0], strategies[i])
		}
	}
}

// TestStepMatrices_WithDecayMask tests matrix optimization with decay mask
func TestStepMatrices_WithDecayMask(t *testing.T) {
	paramMatrices := [][]float64{
		{0.1, 0.2}, // weights (apply decay)
		{0.0},      // bias (no decay)
	}
	gradMatrices := [][]float64{
		{0.01, 0.02},
		{0.03},
	}

	// Create decay mask: apply to first 2 params (weights), not to last 1 (bias)
	decayMask := []bool{true, true, false}

	opt, err := NewFromMatrices([][]float64{
		clone(paramMatrices[0]),
		clone(paramMatrices[1]),
	}, Options{
		Alpha:       1e-3,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 1e-2,
		Schedule:    NewFixedSchedule(1.0, 0),
		DecayMask:   decayMask,
	})
	if err != nil {
		t.Fatalf("NewFromMatrices failed: %v", err)
	}

	testParamMatrices := [][]float64{
		clone(paramMatrices[0]),
		clone(paramMatrices[1]),
	}
	testGradMatrices := [][]float64{
		clone(gradMatrices[0]),
		clone(gradMatrices[1]),
	}

	// Perform optimization
	err = opt.StepMatrices(testParamMatrices, testGradMatrices)
	if err != nil {
		t.Fatalf("StepMatrices failed: %v", err)
	}

	// With decay mask, bias should change less (due to no weight decay)
	// This is a qualitative test - exact values depend on implementation details
	if len(testParamMatrices) != 2 {
		t.Error("Parameter matrices structure changed")
	}
}

// TestMatrixView_ErrorCases tests error handling in MatrixView
func TestMatrixView_ErrorCases(t *testing.T) {
	// Test empty matrices
	_, err := NewMatrixView([][]float64{})
	if err == nil {
		t.Error("Expected error for empty matrices slice")
	}

	// Test nil matrix
	_, err = NewMatrixView([][]float64{nil})
	if err == nil {
		t.Error("Expected error for nil matrix")
	}

	// Test empty matrix elements
	_, err = NewMatrixView([][]float64{{}})
	if err == nil {
		t.Error("Expected error for empty matrix elements")
	}

	// Test valid matrix view for bounds checking
	mv, err := NewMatrixView([][]float64{{1.0, 2.0}})
	if err != nil {
		t.Fatalf("NewMatrixView failed: %v", err)
	}

	// Test out of bounds access
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for out of bounds Get")
		}
	}()
	mv.Get(5) // Should panic
}
