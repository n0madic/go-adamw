package adamw

import (
	"errors"
)

// MatrixView provides a flat view of [][]float64 matrices without copying data.
// It allows treating matrix data as a single flat slice while maintaining
// the original matrix structure for efficient access patterns.
type MatrixView struct {
	matrices [][]float64
	totalLen int
	offsets  []int // cumulative offsets for each matrix
}

// NewMatrixView creates a MatrixView from a slice of matrices.
// The matrices can have different dimensions, and the view provides
// unified access to all elements as if they were in a single flat slice.
func NewMatrixView(matrices [][]float64) (*MatrixView, error) {
	if len(matrices) == 0 {
		return nil, errors.New("matrices slice cannot be empty")
	}

	totalLen := 0
	offsets := make([]int, len(matrices)+1)

	for i, matrix := range matrices {
		if matrix == nil {
			return nil, errors.New("matrix cannot be nil")
		}
		offsets[i] = totalLen
		totalLen += len(matrix)
	}
	offsets[len(matrices)] = totalLen

	if totalLen == 0 {
		return nil, errors.New("matrices cannot be empty")
	}

	return &MatrixView{
		matrices: matrices,
		totalLen: totalLen,
		offsets:  offsets,
	}, nil
}

// Len returns the total number of elements across all matrices
func (mv *MatrixView) Len() int {
	return mv.totalLen
}

// Get returns the element at the given flat index
func (mv *MatrixView) Get(flatIndex int) float64 {
	if flatIndex < 0 || flatIndex >= mv.totalLen {
		panic("index out of bounds")
	}

	// Find which matrix contains this index
	matrixIndex := 0
	for i := 1; i < len(mv.offsets); i++ {
		if flatIndex < mv.offsets[i] {
			matrixIndex = i - 1
			break
		}
	}

	// Convert to local index within the matrix
	localIndex := flatIndex - mv.offsets[matrixIndex]
	matrix := mv.matrices[matrixIndex]

	// Return the element at the local index
	return matrix[localIndex]
}

// Set updates the element at the given flat index
func (mv *MatrixView) Set(flatIndex int, value float64) {
	if flatIndex < 0 || flatIndex >= mv.totalLen {
		panic("index out of bounds")
	}

	// Find which matrix contains this index
	matrixIndex := 0
	for i := 1; i < len(mv.offsets); i++ {
		if flatIndex < mv.offsets[i] {
			matrixIndex = i - 1
			break
		}
	}

	// Convert to local index within the matrix
	localIndex := flatIndex - mv.offsets[matrixIndex]
	matrix := mv.matrices[matrixIndex]

	// Set the element at the local index
	matrix[localIndex] = value
}

// ToFlat creates a flat slice copy of all matrix data.
// This creates a copy and should be used sparingly. Consider using ForEach/ForEachMutable
// for better performance when you don't need a separate flat slice.
func (mv *MatrixView) ToFlat() []float64 {
	flat := make([]float64, mv.totalLen)
	index := 0
	for _, matrix := range mv.matrices {
		copy(flat[index:index+len(matrix)], matrix)
		index += len(matrix)
	}
	return flat
}

// CopyFromFlat copies data from a flat slice back to the matrices.
// The flat slice must have exactly the same length as the total matrix size.
// This method involves copying data. Consider using Set/ForEachMutable for better performance.
func (mv *MatrixView) CopyFromFlat(flat []float64) error {
	if len(flat) != mv.totalLen {
		return errors.New("flat slice length must match total matrix size")
	}

	index := 0
	for _, matrix := range mv.matrices {
		copy(matrix, flat[index:index+len(matrix)])
		index += len(matrix)
	}
	return nil
}

// ForEach applies a function to each element with its flat index.
// This is efficient for operations that need both the index and value.
func (mv *MatrixView) ForEach(fn func(index int, value float64)) {
	flatIndex := 0
	for _, matrix := range mv.matrices {
		for _, value := range matrix {
			fn(flatIndex, value)
			flatIndex++
		}
	}
}

// ForEachMutable applies a function to each element that can modify the value.
// This is efficient for in-place operations on all elements.
func (mv *MatrixView) ForEachMutable(fn func(index int, value *float64)) {
	flatIndex := 0
	for _, matrix := range mv.matrices {
		for i := range matrix {
			fn(flatIndex, &matrix[i])
			flatIndex++
		}
	}
}

// Matrices returns the underlying matrices (read-only access recommended)
func (mv *MatrixView) Matrices() [][]float64 {
	return mv.matrices
}
