
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import (
	"math"
	"math/cmplx"
	"fmt"
	"errors"
)

// A column-major matrix backed by a flat array of all elements.
type FloatMatrix struct {
	dimensions
	// flattened matrix data. elements[i*step+j] is col i, row j
	elements []float64
}


// Create a column-major matrix from a flat array of elements. Assumes
// values are in column-major order.
func FloatNew(rows, cols int, elements []float64) *FloatMatrix {
	e := make([]float64, rows*cols)
	copy(e, elements)
	return makeFloatMatrix(rows, cols, e)
}

// Create a column major vector from an array of elements. Shorthand for
// call MakeMatrix(len(elems), 1, elems).
func FloatVector(elements []float64) *FloatMatrix {
	rows := len(elements)
	e := make([]float64, rows)
	copy(e, elements)
	return makeFloatMatrix(rows, 1, e)
}

// Create a singleton matrix from float value. Shorthand for calling
// MakeMatrix(1, 1, value-array-of-length-one).
func FloatValue(value float64) *FloatMatrix {
	e := make([]float64, 1)
	e[0] = value
	return makeFloatMatrix(1, 1, e)
}

// Create random matrix with elements from [0.0, 1.0).
func FloatRandom(rows, cols int, nonNeg bool) *FloatMatrix {
	A := FloatZeros(rows, cols)
	for i, _ := range A.elements {
		A.elements[i] = uniformFloat64(nonNeg)
	}
	return A
}

// Create symmetric n by n random  matrix with elements from [0.0, 1.0).
func FloatRandomSymmetric(n int, nonNeg bool) *FloatMatrix {
	A := FloatZeros(n, n)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			val := uniformFloat64(nonNeg)
			A.Set(i, j, val)
			if i != j {
				A.Set(j, i, val)
			}
		}
	}
	return A
}

// Create a column-major matrix from a array of arrays. Parameter rowOrder
// indicates if data is array of rows (true) or array of columns (false).
func FloatMatrixStacked(data [][]float64, rowOrder bool) *FloatMatrix {
	var rows, cols int
	if rowOrder {
		rows = len(data)
		cols = len(data[0])
	} else {
		cols = len(data)
		rows = len(data[0])
	}
	elements := make([]float64, rows*cols)
	if rowOrder {
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
	return makeFloatMatrix(rows, cols, elements)
}

// Create a new matrix from a list of matrices. New matrix has dimension (N, colmax),
// where N is sum of row counts of argument matrices and colmax is the largest column
// count of matrices.
func FloatCombined(mlist... *FloatMatrix) *FloatMatrix {
	maxc := 0
	maxr := 0
	N := 0
	for _, m := range mlist {
		m, n := m.Size()
		N += m
		if m > maxr {
			maxr = m
		}
		if n > maxc {
			maxc = n
		}
	}
	M := FloatZeros(N, maxc)
	row := 0
	for _, m := range mlist {
		M.SetSubMatrix(row, 0, m)
		row += m.Rows()
	}
	return M
}

// Create new zero filled matrix.
func FloatZeros(rows, cols int) *FloatMatrix {
	A := makeFloatMatrix(rows, cols, make([]float64, rows*cols))
	return A
}

// Create new matrix initialized to one.
func FloatOnes(rows, cols int) *FloatMatrix {
	return FloatNumbers(rows, cols, 1.0)
}

// Create new matrix initialized to value.
func FloatNumbers(rows, cols int, value float64) *FloatMatrix {
	A := FloatZeros(rows, cols)
	for k, _ := range A.elements {
		A.elements[k] = value
	}
	return A
}

// Create new identity matrix. Row count must equal column count.
func FloatIdentity(rows int) *FloatMatrix {
	A := FloatZeros(rows, rows)
	step := A.LeadingIndex()
	for k := 0; k < rows; k++ {
		A.elements[k*step+k] = 1.0
	}
	return A
}

func FloatDiagonal(rows int, val float64) *FloatMatrix {
	A := FloatZeros(rows, rows)
	step := A.LeadingIndex()
	for k := 0; k < rows; k++ {
		A.elements[k*step+k] = val
	}
	return A
}

// Return the flat column-major element array.
func (A *FloatMatrix) FloatArray() []float64 {
	if A == nil {
		return nil
	}
	return A.elements
}

// Return nil for complex array 
func (A *FloatMatrix) ComplexArray() []complex128 {
	return nil
}

// Return the first element column-major element array.
func (A *FloatMatrix) Float() float64 {
	if A == nil {
		return math.NaN()
	}
	return A.elements[0]
}

// Return Nan for complex singleton.
func (A *FloatMatrix) Complex() complex128 {
	return cmplx.NaN()
}

// Return false for float matrix.
func (A *FloatMatrix) IsComplex() bool {
	return false
}

// Test if parameter matrices are of same type as self.
func (A *FloatMatrix) EqualTypes(mats ...Matrix) bool {
loop:
	for _, m := range mats {
		if m == nil { continue loop }
		switch m.(type) {
		case *FloatMatrix:	// of same type, NoOp
		default:		// all others fail.
			return false
		}
	}
	return true
}

// Get the element in the i'th row and j'th column.
func (A *FloatMatrix) Get(i int, j int) (val float64) {
	step := A.LeadingIndex()
	//val = A.elements[j*step:j*step+A.Cols()][i]
	val = A.elements[j*step+i]
	return
}

// Get i'th element in column-major ordering
func (A *FloatMatrix) GetAt(i int) float64 {
	if i < 0 {
		i = A.NumElements() + i
	}
	i %= A.NumElements()
	return A.elements[i]
}

func (A *FloatMatrix) GetIndex(i int) float64 {
	return A.GetAt(i)
}

// Get values for indexed elements. 
func (A *FloatMatrix) GetIndexes(indexes []int) []float64 {
	vals := make([]float64, len(indexes))
	for i, k := range indexes {
		if k < 0 {
			k = A.NumElements() + k
		}
		k %= A.NumElements()
		vals[i] = A.elements[k]
	}
	return vals
}

// Get copy of i'th row. Row elements are copied to vals array. 
// Returns the array. If vals array is too small new slice is allocated and 
// returned with row elements.
func (A *FloatMatrix) GetRow(i int, vals []float64) []float64 {
	if cap(vals) < A.Cols() {
		vals = make([]float64, A.Cols())
	}
	step := A.LeadingIndex()
	for j := 0; j < A.Cols(); j++ {
		vals[j] = A.elements[j*step+i]
	}
	return vals
}

// Get copy of i'th row. Return parameter matrix. If vec is too small 
// reallocate new vector and return it.
func (A *FloatMatrix) GetRowMatrix(i int, vec *FloatMatrix) *FloatMatrix {
	if vec == nil || vec.NumElements() < A.Cols() {
		vec = FloatZeros(A.Cols(), 1)
	}
	step := A.LeadingIndex()
	ar := vec.FloatArray()
	for j := 0; j < A.Cols(); j++ {
		ar[j] = A.elements[j*step+i]
	}
	return vec
}

// Get copy of i'th column. See GetRow.
func (A *FloatMatrix) GetColumnMatrix(i int, vec *FloatMatrix) *FloatMatrix {
	if vec == nil || vec.NumElements() < A.Rows() {
		vec = FloatZeros(A.Rows(), 1)
	}
	step := A.LeadingIndex()
	ar := vec.FloatArray()
	for j := 0; j < A.Rows(); j++ {
		ar[j] = A.elements[i*step+j]
	}
	return vec
}

// Get copy of i'th column. See GetRow.
func (A *FloatMatrix) GetColumn(i int, vec []float64) []float64 {
	if cap(vec) < A.Rows() {
		vec = make([]float64, A.Rows())
	}
	step := A.LeadingIndex()
	for j := 0; j < A.Rows(); j++ {
		vec[j] = A.elements[i*step+j]
	}
	return vec
}

// Get a slice from the underlying storage array. Changing entries
// in the returned slices changes the matrix. Be carefull with this.
func (A *FloatMatrix) GetSlice(start, end int) []float64 {
	if start < 0 {
		start = 0
	}
	if end > A.NumElements() {
		end = A.NumElements()
	}
	return A.elements[start:end]
}

// Set the element in the i'th row and j'th column to val.
func (A *FloatMatrix) Set(i int, j int, val float64) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Rows() + i
	}
	if j < 0 {
		j = A.Cols() + j
	}
	i %= A.Rows()
	j %= A.Cols()
	A.elements[j*step+i] = val
}

// Set value of singleton matrix.
func (A *FloatMatrix) SetValue(val float64) {
	A.elements[0] = val
}

// Set i'th element in column-major ordering. If i < 0 then i = A.NumElements() + i.
// Last element of 
func (A *FloatMatrix) SetAt(i int, v float64) {
	if i < 0 {
		i = A.NumElements() + i
	}
	i %= A.NumElements()
	A.elements[i] = v
}

func (A *FloatMatrix) SetIndex(i int, v float64) {
	A.SetAt(i, v)
}

// Set values of indexed elements. 
func (A *FloatMatrix) SetIndexes(indexes []int, values []float64) {
	for i, k := range indexes {
		if i >= len(values) {
			break
		}
		if k < 0 {
			k = A.NumElements() + k
		}
		k %= A.NumElements()
		A.elements[k] = values[i]
	}
}

// Set values of i'th row.
func (A *FloatMatrix) SetRow(i int, vals []float64) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Rows() + i
	}
	i %= A.Rows()
	for j := 0; j < A.Cols(); j++ {
		A.elements[j*step+i] = vals[j]
	}
}

// Set values of i'th row. Matrix vals is either (A.Cols(), 1) or (1, A.Cols()) matrix.
func (A *FloatMatrix) SetRowMatrix(i int, vals *FloatMatrix) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Rows() + i
	}
	i %= A.Rows()
	for j := 0; j < A.Cols(); j++ {
		A.elements[j*step+i] = vals.elements[j]
	}
}


// Set values of i'th column.
func (A *FloatMatrix) SetColumn(i int, vals []float64) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Cols() + i
	}
	i %= A.Cols()
	for j := 0; j < A.Rows(); j++ {
		A.elements[i*step+j] = vals[j]
	}
}

// Set values of i'th column. Matrix vals is either (A.Rows(), 1) or (1, A.Rows()) matrix.
func (A *FloatMatrix) SetColumnMatrix(i int, vals *FloatMatrix) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Cols() + i
	}
	i %= A.Cols()
	for j := 0; j < A.Rows(); j++ {
		A.elements[i*step+j] = vals.elements[j]
	}
}


// Set values for sub-matrix starting at (row, col). If row+mat.Rows() greater than
// A.Rows() or col+mat.Cols() greater than A.Cols() matrix A is not changed.
func (A *FloatMatrix) SetSubMatrix(row, col int, mat *FloatMatrix) error {
	r, c := mat.Size()
	if r + row > A.Rows() || c + col > A.Cols() {
		s := fmt.Sprintf("(%d+%d, %d+%d) > (%d,%d)\n", r, row, c, col, A.Rows(), A.Cols())
		return errors.New(s)
	}
	for i := 0; i < r; i++  {
		for j := 0; j < c; j++ {
			A.Set(row+i, col+j, mat.Get(i, j))
		}
	}
	return nil
}

// Create a copy of matrix.
func (A *FloatMatrix) Copy() (B *FloatMatrix) {
	B = new(FloatMatrix)
	B.elements = make([]float64, A.NumElements())
	B.SetSize(A.Rows(), A.Cols())
	copy(B.elements, A.elements)
	return 
}

func (A *FloatMatrix) MakeCopy() Matrix {
	return A.Copy()
}

// Copy and transpose matrix. Returns new matrix.
func (A *FloatMatrix) Transpose() *FloatMatrix {
	rows := A.Rows()
	cols := A.Cols()
	newelems := transposeFloatArray(rows, cols, A.elements)
	return makeFloatMatrix(cols, rows, newelems)
}

// Transpose matrix in place. Returns original.
func (A *FloatMatrix) TransposeInPlace() *FloatMatrix {
	rows := A.Rows()
	cols := A.Cols()
	newelems := transposeFloatArray(rows, cols, A.elements)
	A.SetSize(cols, rows)
	// not really in-place, but almost :)
	copy(A.elements, newelems)
	return A
}


// Transpose a column major data array.
func transposeFloatArray(rows, cols int, data []float64) []float64 {
	newelems := make([]float64, rows*cols)
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
func makeFloatMatrix(rows, cols int, elements []float64) *FloatMatrix {
	A := new(FloatMatrix)
	A.SetSize(rows, cols)
	//A.rows = rows
	//A.cols = cols
	//A.step = rows
	A.elements = elements
	return A
}

// Local Variables:
// tab-width: 4
// End:
