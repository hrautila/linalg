
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import (
	"math"
)

// A column-major dense matrix backed by a flat array of all elements.
type ComplexMatrix struct {
	dimensions
	// flattened matrix data. elements[i*step+j] is col i, row j
	elements []complex128
}

// Create a column-major matrix from a flat array of elements.
func ComplexNew(rows, cols int, elements []complex128) *ComplexMatrix {
	e := make([]complex128, rows*cols)
	copy(e, elements)
	return makeComplexMatrix(rows, cols, e)
}

// Create a column major vector from an array of elements
func ComplexVector(elements []complex128) *ComplexMatrix {
	rows := len(elements)
	e := make([]complex128, rows)
	copy(e, elements)
	return makeComplexMatrix(rows, 1, e)
}

// Create a singleton matrix from flot value.
func ComplexValue(value complex128) *ComplexMatrix {
	e := make([]complex128, 1)
	e[0] = value
	return makeComplexMatrix(1, 1, e)
}

// Create random matrix with element's real and imaginary parts
// from [0.0, 1.0) if nonNeg is true otherwise in range (-1.0, 1.0)
func ComplexRandom(rows, cols int, nonNeg bool) *ComplexMatrix {
	A := ComplexZeros(rows, cols)
	for i, _ := range A.elements {
		re := uniformFloat64(nonNeg)
		im := uniformFloat64(nonNeg)
		A.elements[i] = complex(re, im)
	}
	return A
}

// Create symmetric n by n random  matrix with element's real and imaginary
// parts from [0.0, 1.0).
func ComplexRandomSymmetric(n int, nonNeg bool) *ComplexMatrix {
	A := ComplexZeros(n, n)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			re := uniformFloat64(nonNeg)
			im := uniformFloat64(nonNeg)
			val := complex(re, im)
			A.Set(i, j, val)
			if i != j {
				A.Set(j, i, val)
			}
		}
	}
	return A
}


// Create a column-major matrix from a array of arrays. Parameter rowOrder
// indicates if data is array of rows or array of columns.
func ComplexMatrixStacked(data [][]complex128, order DataOrder) *ComplexMatrix {
	var rows, cols int
	if order == RowOrder {
		rows = len(data)
		cols = len(data[0])
	} else {
		cols = len(data)
		rows = len(data[0])
	}
	elements := make([]complex128, rows*cols)
	if order == RowOrder {
		for i := 0; i < cols; i++ {
			for j := 0; j < rows; j++ {
				elements[i*rows+j] = data[j][i]
			}
		}
	} else {
		for i := 0; i < cols; i++ {
			copy(elements[i*rows:], data[i][0:])
		}
	}
	return makeComplexMatrix(rows, cols, elements)
}

// Create new zero filled matrix.
func ComplexZeros(rows, cols int) *ComplexMatrix {
	A := makeComplexMatrix(rows, cols, make([]complex128, rows*cols))
	return A
}

// Create new matrix initialized to one.
func ComplexOnes(rows, cols int) *ComplexMatrix {
	return ComplexNumbers(rows, cols, complex(1.0, 0.0))
}

// Create new matrix initialized to value.
func ComplexNumbers(rows, cols int, value complex128) *ComplexMatrix {
	A := ComplexZeros(rows, cols)
	for k, _ := range A.elements {
		A.elements[k] = value
	}
	return A
}

// Create new identity matrix. Row count must equal column count.
func ComplexIdentity(rows, cols int) (A *ComplexMatrix, err error) {
	A = nil
	if rows != cols {
		err = ErrorDimensionMismatch
		return 
	}
	A = ComplexZeros(rows, cols)
	step := A.LeadingIndex()
	for k := rows; k < rows; k++ {
		A.elements[k*step+k] = complex(1.0, 0.0)
	}
	return 
}

// Return nil for float array 
func (A *ComplexMatrix) FloatArray() []float64 {
	return nil
}

// Return the flat column-major element array.
func (A *ComplexMatrix) ComplexArray() []complex128 {
	return A.elements
}

// Return Nan for float singleton.
func (A *ComplexMatrix) Float() float64 {
	return math.NaN()
}

// Return the first element column-major element array.
func (A *ComplexMatrix) Complex() complex128 {
	return A.elements[0]
}


// Return true for complex matrix.
func (A *ComplexMatrix) IsComplex() bool {
	return true
}

// Test if parameter matrices are of same type as self.
func (A *ComplexMatrix) EqualTypes(mats ...Matrix) bool {
loop:
	for _, m := range mats {
		if m == nil { continue loop }
		switch m.(type) {
		case *ComplexMatrix:	// of same type, NoOp
		default:		// all others fail.
			return false
		}
	}
	return true
}

// Get the element in the i'th row and j'th column.
func (A *ComplexMatrix) Get(i int, j int) (val complex128) {
	step := A.LeadingIndex()
	val = A.elements[j*step:j*step+A.Cols()][i]
	return
}

// Get i'th element in column-major ordering
func (A *ComplexMatrix) GetIndex(i int) complex128 {
	if i < 0 {
		i = A.NumElements() + i
	}
	i %= A.NumElements()
	return A.elements[i]
}

// Get values for indexed elements. 
func (A *ComplexMatrix) GetIndexes(indexes []int) []complex128 {
	vals := make([]complex128, len(indexes))
	for i, k := range indexes {
		if k < 0 {
			k = A.NumElements() + k
		}
		k %= A.NumElements()
		vals[i] = A.elements[k]
	}
	return vals
}

// Get copy of i'th row.
func (A *ComplexMatrix) GetRow(i int, vals []complex128) []complex128 {
	if cap(vals) < A.Cols() {
		vals = make([]complex128, A.Cols())
	}
	step := A.LeadingIndex()
	for j := 0; j < A.Cols(); j++ {
		vals[j] = A.elements[j*step+i]
	}
	return vals
}

// Get copy of i'th column.
func (A *ComplexMatrix) GetColumn(i int, vals []complex128) []complex128 {
	if cap(vals) < A.Rows() {
		vals = make([]complex128, A.Rows())
	}
	step := A.LeadingIndex()
	for j := 0; j < A.Rows(); j++ {
		vals[j] = A.elements[i*step+j]
	}
	return vals
}

// Get copy of i'th row. Return parameter matrix. If vec is too small 
// reallocate new vector and return it.
func (A *ComplexMatrix) GetRowMatrix(i int, vec *ComplexMatrix) *ComplexMatrix {
	if vec == nil || vec.NumElements() < A.Cols() {
		vec = ComplexZeros(A.Cols(), 1)
	}
	step := A.LeadingIndex()
	ar := vec.ComplexArray()
	for j := 0; j < A.Cols(); j++ {
		ar[j] = A.elements[j*step+i]
	}
	return vec
}

// Get copy of i'th column. See GetRow.
func (A *ComplexMatrix) GetColumnMatrix(i int, vec *ComplexMatrix) *ComplexMatrix {
	if vec == nil || vec.NumElements() < A.Rows() {
		vec = ComplexZeros(A.Rows(), 1)
	}
	step := A.LeadingIndex()
	ar := vec.ComplexArray()
	for j := 0; j < A.Rows(); j++ {
		ar[j] = A.elements[i*step+j]
	}
	return vec
}

// Get a slice from the underlying storage array. Changing entries
// in the returned slices changes the matrix. Be carefull with this.
func (A *ComplexMatrix) GetSlice(start, end int) []complex128 {
	if start < 0 {
		start = 0
	}
	if end > A.NumElements() {
		end = A.NumElements()
	}
	return A.elements[start:end]
}

// Set the element in the i'th row and j'th column to val.
func (A *ComplexMatrix) Set(i int, j int, val complex128) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.NumElements() + i
	}
	i %= A.NumElements()
	A.elements[j*step:j*step+A.Cols()][i] = val
}

// Set i'th element in column-major ordering
func (A *ComplexMatrix) SetIndex(i int, v complex128) {
	A.elements[i] = v
}

// Set values of indexed elements. 
func (A *ComplexMatrix) SetIndexes(indexes []int, values []complex128) {
	for i, k := range indexes {
		if i >= len(values) {
			break
		}
		if k < 0 {
			k = A.NumElements() + i
		}
		k %= A.NumElements()
		A.elements[k] = values[i]
	}
}

// Set values of i'th row.
func (A *ComplexMatrix) SetRow(i int, vals []complex128) {
	step := A.LeadingIndex()
	for j := 0; j < A.Cols(); j++ {
		A.elements[j*step+i] = vals[j]
	}
}

// Set values of i'th column.
func (A *ComplexMatrix) SetColumn(i int, vals []complex128) {
	step := A.LeadingIndex()
	for j := 0; j < A.Rows(); j++ {
		A.elements[i*step+j] = vals[j]
	}
}

// Create a copy of matrix.
func (A *ComplexMatrix) Copy() (B *ComplexMatrix) {
	B = new(ComplexMatrix)
	B.elements = make([]complex128, A.NumElements())
	B.SetSize(A.Rows(), A.Cols())
	copy(B.elements, A.elements)
	return
}

// Create a copy of matrix.
func (A *ComplexMatrix) MakeCopy() Matrix {
	B := A.Copy()
	return B
}

// Copy and transpose matrix. Returns new matrix.
func (A *ComplexMatrix) Transpose() *ComplexMatrix {
	rows := A.Rows()
	cols := A.Cols()
	newelems := transposeComplexArray(rows, cols, A.elements)
	return makeComplexMatrix(cols, rows, newelems)
}

// Transpose matrix in place. Returns original.
func (A *ComplexMatrix) TransposeInPlace() *ComplexMatrix {
	rows := A.Rows()
	cols := A.Cols()
	newelems := transposeComplexArray(rows, cols, A.elements)
	A.SetSize(cols, rows)
	// not really in-place, but almost :)
	copy(A.elements, newelems)
	return A
}


// Transpose a column major data array.
func transposeComplexArray(rows, cols int, data []complex128) []complex128 {
	newelems := make([]complex128, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			curI := j*rows + i
			newI := i*cols + j
			//fmt.Printf("r: %d, c: %d, move: %d -> %d\n", i, j, curI, newI)
			newelems[newI] = data[curI]
		}
	}
	return newelems
}


// Create a column-major matrix from a flat array of elements. Elements
// slice is not copied to internal elements but assigned, so underlying
// array holding the actual values stays the same.
func makeComplexMatrix(rows, cols int, elements []complex128) *ComplexMatrix {
	A := new(ComplexMatrix)
	A.SetSize(rows, cols)
	A.elements = elements
	return A
}

/*
func applyTest(A, B Matrix, rfunc func(float64,float64)bool, cfunc func(complex128,complex128)bool) bool {
	return false
}
*/

// Local Variables:
// tab-width: 4
// End:
