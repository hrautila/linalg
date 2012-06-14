
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

// Package matrix implements column major matrices.
package matrix


// Minimal interface for linear algebra packages BLAS/LAPACK
type Matrix interface {
	// The number of rows in this matrix.
	Rows() int
	// The number of columns in this matrix.
	Cols() int
	// The number of elements in this matrix.
	NumElements() int
	// Returns underlying float64 array for BLAS/LAPACK routines. Returns nil
	// if matrix is complex128 valued.
	FloatArray() []float64
	// Returns underlying complex128 array for BLAS/LAPACK routines. Returns nil
	// if matrix is float64 valued matrix.
	ComplexArray() []complex128
	// Returns true if matrix is complex valued. False otherwise.
	IsComplex() bool
	// For all float valued matrices return the value of A[0,0]. Returns NaN
	// if not float valued.
	FloatValue() float64
	// For all complex valued matrices return the value of A[0,0]. Returns
	// NaN if not complex valued.
	ComplexValue() complex128
	// Matrix in string format.
	String() string
	// Make a copy  and return as Matrix interface type.
	MakeCopy() Matrix
	// Match size. Return true if equal.
	SizeMatch(int, int) bool
	// Get matrix size. Return pair (rows, cols).
	Size() (int, int)
	// Test for type equality.
	EqualTypes(...Matrix) bool
}

//type Index struct {
//	Row int
//	Col int
//}

// Matrix dimensions, rows, cols and leading index. For column major matrices 
// leading index is equal to row count.
type dimensions struct {
	rows int
	cols int
	// actual offset between leading index
	step int
}

// Return number of rows.
func (A *dimensions) Rows() int {
	return A.rows
}

// Return number of columns.
func (A *dimensions) Cols() int {
	return A.cols
}

// Return number of size of the matrix as rows, cols pair.
func (A *dimensions) Size() (int, int) {
	return A.rows, A.cols
}

// Set dimensions. Does not affect element allocations.
func (A *dimensions) SetSize(nrows, ncols int) {
	A.rows = nrows
	A.cols = ncols
	A.step = A.rows
}

// Return the leading index size. Column major matrices it is row count.
func (A *dimensions) LeadingIndex() int {
	return A.step
}

// Return total number of elements.
func (A *dimensions) NumElements() int {
	return A.rows * A.cols
}


// Return true if size of A is equal to size of B.
func (A *dimensions) SizeMatch(rows, cols int) bool {
	return A != nil && A.rows == rows && A.cols == cols
}


// Local Variables:
// tab-width: 4
// End:
