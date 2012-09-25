
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import (
	"math"
	"math/cmplx"
	"math/rand"
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

// Create random matrix with elements from [0.0, 1.0) uniformly distributed..
func FloatUniform(rows, cols int) *FloatMatrix {
	A := FloatZeros(rows, cols)
	for i, _ := range A.elements {
		A.elements[i] = rand.Float64()
	}
	return A
}

// Create random matrix with elements from normal distribution (mean=0.0, stddev=1.0)
func FloatNormal(rows, cols int) *FloatMatrix {
	A := FloatZeros(rows, cols)
	for i, _ := range A.elements {
		A.elements[i] = rand.NormFloat64()
	}
	return A
}

// Create symmetric n by n random  matrix with elements from [0.0, 1.0).
func FloatUniformSymmetric(n int) *FloatMatrix {
	A := FloatZeros(n, n)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			val := rand.Float64()
			A.SetAt(i, j, val)
			if i != j {
				A.SetAt(j, i, val)
			}
		}
	}
	return A
}

// Create symmetric n by n random  matrix with elements from normal distribution.
func FloatNormalSymmetric(n int) *FloatMatrix {
	A := FloatZeros(n, n)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			val := rand.NormFloat64()
			A.SetAt(i, j, val)
			if i != j {
				A.SetAt(j, i, val)
			}
		}
	}
	return A
}

// Create a column-major matrix from a array of arrays. Parameter order
// indicates if data is array of rows (RowOrder) or array of columns (ColumnOrder).
func FloatMatrixFromTable(data [][]float64, order... DataOrder) *FloatMatrix {
	var rows, cols int
	if len(order) == 0 || order[0] == ColumnOrder {
		cols = len(data)
		rows = len(data[0])
	} else {
		rows = len(data)
		cols = len(data[0])
	}

	if rows*cols == 0 {
		return FloatZeros(rows, cols)
	}
	elements := make([]float64, rows*cols)
	if len(order) == 0 || order[0] == ColumnOrder {
		for i := 0; i < cols; i++ {
			copy(elements[i*rows:], data[i][0:])
		}
	} else {
		for i := 0; i < cols; i++ {
			for j := 0; j < rows; j++ {
				elements[i*rows+j] = data[j][i]
			}
		}
	}
	return makeFloatMatrix(rows, cols, elements)
}

// Create a new matrix from a list of matrices. New matrix has dimension (M, colmax)
// if direction is StackDown, and (rowmax, N) if direction is StackRight. 
// M is sum of row counts of argument matrices and N is sum of column counts of arguments.
// Colmax is the largest column count of matrices and rowmax is the largest row count.
// Return  new matrix and array of submatrix sizes, row counts for StackDown and column
// counts for StackRight
func FloatMatrixStacked(direction Stacking, mlist... *FloatMatrix) (*FloatMatrix, []int) {
	maxc := 0
	maxr := 0
	N := 0
	M := 0
	for _, m := range mlist {
		m, n := m.Size()
		M += m
		N += n
		if m > maxr {
			maxr = m
		}
		if n > maxc {
			maxc = n
		}
	}
	var mat *FloatMatrix
	indexes := make([]int, 0)
	if direction == StackDown {
		mat = FloatZeros(M, maxc)
		row := 0
		for _, m := range mlist {
			mat.SetSubMatrix(row, 0, m)
			indexes = append(indexes, m.Rows())
			row += m.Rows()
		}
	} else {
		mat = FloatZeros(maxr, N)
		col := 0
		for _, m := range mlist {
			mat.SetSubMatrix(0, col, m)
			indexes = append(indexes, m.Cols())
			col += m.Cols()
		}
	}
	return mat, indexes
}

// Create new zero filled matrix.
func FloatZeros(rows, cols int) *FloatMatrix {
	A := makeFloatMatrix(rows, cols, make([]float64, rows*cols))
	return A
}

// Create new matrix initialized to one.
func FloatOnes(rows, cols int) *FloatMatrix {
	return FloatWithValue(rows, cols, 1.0)
}

// Create new matrix initialized to value.
func FloatWithValue(rows, cols int, value float64) *FloatMatrix {
	A := FloatZeros(rows, cols)
	for k, _ := range A.elements {
		A.elements[k] = value
	}
	return A
}

// Create new identity matrix. Row count must equal column count.
func FloatIdentity(rows int) *FloatMatrix {
	return FloatDiagonal(rows, 1.0)
}

// Make a square matrix with diagonal set to values. If len(values) is one
// then all entries on diagonal is set to values[0]. If len(values) is
// greater than one then diagonals are set from the list values starting
// from (0,0) until the diagonal is full or values are exhausted.
func FloatDiagonal(rows int, values ...float64) *FloatMatrix {
	A := FloatZeros(rows, rows)
	step := A.LeadingIndex()
	if len(values) == 1 {
		for k := 0; k < rows; k++ {
			A.elements[k*step+k] = values[0]
		}
	} else {
		for k:= 0; k < rows && k < len(values); k++ {
			A.elements[k*step+k] = values[k]
		}
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
func (A *FloatMatrix) GetAt(i int, j int) (val float64) {
	step := A.LeadingIndex()
	//val = A.elements[j*step:j*step+A.Cols()][i]
	if i < 0 {
		i += A.Rows()
	}
	if j < 0 {
		j += A.Cols()
	}
	val = A.elements[j*step+i]
	return
}

// Get elements from column-major indexes. Return new array.
func (A *FloatMatrix) Get(indexes... int) []float64 {
	vals := make([]float64, 0)
	N := A.NumElements()
	for _, k := range indexes {
		if k < 0 {
			k += N
		}
		vals = append(vals, A.elements[k])
	}
	return vals
}

func (A *FloatMatrix) GetIndex(i int) float64 {
	return A.Get(i)[0]
}

// Get values for indexed elements. 
func (A *FloatMatrix) GetIndexes(indexes []int) []float64 {
	return A.Get(indexes...)
}

// Get copy of i'th row. Row elements are copied to vals array. 
// Returns the array. If vals array is too small new slice is allocated and 
// returned with row elements.
func (A *FloatMatrix) GetRowArray(i int, vals []float64) []float64 {
	if vals == nil || cap(vals) < A.Cols() {
		vals = make([]float64, A.Cols())
	}
	step := A.LeadingIndex()
	if i < 0 {
		i += A.Rows()
	}
	for j := 0; j < A.Cols(); j++ {
		vals[j] = A.elements[j*step+i]
	}
	return vals
}

// Get copy of i'th row. Return parameter matrix. If vec is too small 
// reallocate new vector and return it.
func (A *FloatMatrix) GetRow(i int, vec *FloatMatrix) *FloatMatrix {
	if vec == nil || vec.NumElements() < A.Cols() {
		vec = FloatZeros(1, A.Cols())
	}
	step := A.LeadingIndex()
	ar := vec.FloatArray()
	if i < 0 {
		i += A.Rows()
	}
	for j := 0; j < A.Cols(); j++ {
		ar[j] = A.elements[j*step+i]
	}
	return vec
}

// Get copy of i'th column. See GetRow.
func (A *FloatMatrix) GetColumn(i int, vec *FloatMatrix) *FloatMatrix {
	if vec == nil || vec.NumElements() < A.Rows() {
		vec = FloatZeros(A.Rows(), 1)
	}
	step := A.LeadingIndex()
	ar := vec.FloatArray()
	if i < 0 {
		i += A.Cols()
	}
	for j := 0; j < A.Rows(); j++ {
		ar[j] = A.elements[i*step+j]
	}
	return vec
}

// Get copy of i'th column. See GetRow.
func (A *FloatMatrix) GetColumnArray(i int, vec []float64) []float64 {
	if cap(vec) < A.Rows() {
		vec = make([]float64, A.Rows())
	}
	step := A.LeadingIndex()
	if i < 0 {
		i += A.Cols()
	}
	for j := 0; j < A.Rows(); j++ {
		vec[j] = A.elements[i*step+j]
	}
	return vec
}


// Set the element in the i'th row and j'th column to val.
func (A *FloatMatrix) SetAt(i, j int, val float64) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Rows() + i
	}
	if j < 0 {
		j = A.Cols() + j
	}
	A.elements[j*step+i] = val
}

// Set value of singleton matrix.
func (A *FloatMatrix) SetValue(val float64) {
	A.elements[0] = val
}

// Set element values in column-major ordering. Negative indexes are relative 
// to the last element of the matrix.
func (A *FloatMatrix) Set(val float64, indexes... int) {
	N := A.NumElements()
	for _, i := range indexes {
		if i < 0 {
			i += N
		}
		A.elements[i] = val
	}
}

// Set value of i'th element. 
func (A *FloatMatrix) SetIndex(i int, val float64) {
	A.Set(val, i)
}

// Set values of indexed elements. 
func (A *FloatMatrix) SetIndexes(indexes []int, values []float64) {
	for i, k := range indexes {
		if i >= len(values) {
			break
		}
		if k < 0 {
			k += A.NumElements()
		}
		A.elements[k] = values[i]
	}
}

// Set values of i'th row.
func (A *FloatMatrix) SetRowArray(i int, vals []float64) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Rows() + i
	}
	for j := 0; j < A.Cols(); j++ {
		A.elements[j*step+i] = vals[j]
	}
}

// Set values on i'th row  of columns pointed with cols array. 
// For all j in indexes: A[i,j] = vals[k] where k is j's index in indexes array.
func (A *FloatMatrix) SetAtRowArray(i int, cols []int, vals []float64) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Rows() + i
	}
	for k, j := range cols {
		if j < 0 {
			j += A.Cols()
		}
		A.elements[j*step+i] = vals[k]
	}
}

// Set values of i'th row. Matrix vals is either (A.Cols(), 1) or (1, A.Cols()) matrix.
func (A *FloatMatrix) SetRow(i int, vals *FloatMatrix) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Rows() + i
	}
	for j := 0; j < A.Cols(); j++ {
		A.elements[j*step+i] = vals.elements[j]
	}
}


// Set values  on i'th row of columns pointed with cols array. 
// For all j in indexes: A[i,j] = vals[j]. Matrix vals is either (A.Cols(),1) or
// (1, A.Cols()) matrix.
func (A *FloatMatrix) SetAtRow(i int, cols []int, vals *FloatMatrix) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Rows() + i
	}
	for _, j := range cols {
		if j < 0 {
			j += A.Cols()
		}
		A.elements[j*step+i] = vals.elements[j]
	}
}


// Set values of i'th column.
func (A *FloatMatrix) SetColumnArray(i int, vals []float64) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Cols() + i
	}
	for j := 0; j < A.Rows(); j++ {
		A.elements[i*step+j] = vals[j]
	}
}

// Set values on i'th column of rows pointed by rows array. It assumes
// that len(rows) <= len(vals).
func (A *FloatMatrix) SetAtColumnArray(i int, rows []int, vals []float64) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Cols() + i
	}
	for k, j := range rows {
		if j < 0 {
			j += A.Rows()
		}
		A.elements[i*step+j] = vals[k]
	}
}

// Set values of i'th column. Matrix vals is either (A.Rows(), 1) or (1, A.Rows()) matrix.
func (A *FloatMatrix) SetColumn(i int, vals *FloatMatrix) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Cols() + i
	}
	for j := 0; j < A.Rows(); j++ {
		A.elements[i*step+j] = vals.elements[j]
	}
}

// Set values on i'th column of rows pointer by rows array. It assumes
// that  max(rows) < vals.NumElements(). 
func (A *FloatMatrix) SetAtColumn(i int, rows []int, vals *FloatMatrix) {
	step := A.LeadingIndex()
	if i < 0 {
		i = A.Cols() + i
	}
	for _, j := range rows {
		if j < 0 {
			j += A.Rows()
		}
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
			A.SetAt(row+i, col+j, mat.GetAt(i, j))
		}
	}
	return nil
}

// Get sub-matrix starting at (row, col). Sizes parameters define (nrows, ncols) number of
// rows and number of columns. If len(sizes) is zero size is then (nrows, ncols) is
// (Rows()-row, Cols()-col).If len(sizes) is one then (nrows, ncols) is (sizes[0], Cols()-col)
// In all other cases (nrows, ncols) is (sizes[0], sizes[1]). 
// Return nil if nrows+row >= A.Rows() or ncols+col >= A.Cols()
func (A *FloatMatrix) GetSubMatrix(row, col int, sizes ...int) (m *FloatMatrix) {
	var nrows, ncols int = 0, 0
	switch len(sizes) {
	case 0:
		nrows = A.Rows() - row; ncols = A.Cols() - col
	case 1:
		nrows = sizes[0]
		ncols = A.Cols() - col
	default:
		nrows = sizes[0]
		ncols = sizes[1]
	}
	if row + nrows > A.Rows() || col + ncols > A.Cols() {
		return nil
	}
	var colArray []float64 = nil
	m = FloatZeros(nrows, ncols)
	for i := 0; i < ncols; i++  {
		colArray = A.GetColumnArray(col+i, colArray)
		m.SetColumnArray(i, colArray[row:])
	}
	return m
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
