
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import (
	"math"
)


// Make a copy C of A and compute  C *= alpha for all elements in the matrix if list of indexes
// is empty. Otherwise compute C[i] *= alpha for i in indexes array. 
func Scale(A *FloatMatrix, alpha float64, indexes... int) *FloatMatrix {
	C := A.Copy()
	Cr := C.FloatArray()
	if len(indexes) == 0 {
		for k, _ := range Cr {
			Cr[k] *= alpha
		}
	} else  {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			Cr[k] *= alpha
		}
	}
	return C
}

// Make a copy C of A and compute for all k in indexes: C[k] *= values[k].
// Indexes are in column-major order.  Returns a new matrix
func ScaleAt(A *FloatMatrix, values []float64, indexes []int) *FloatMatrix {
	C := A.Copy()
	if len(indexes) == 0 {
		return C
	}
	Cr := C.FloatArray()
	N := A.NumElements()
	for i, k := range indexes {
		if i >= len(values) {
			return C
		}
		if k < 0 {
			k = N + k
		}
		Cr[k] *= values[i]
	}
	return C
}


// Make a copy C of A and compute inverse C[i] = 1.0/C[i]. If indexes is empty calculates for
// all elements. Returns a new matrix.
func Inv(A *FloatMatrix, indexes... int) *FloatMatrix {
	C := A.Copy()
	Cr := C.FloatArray()
	if len(indexes) == 0 {
		for k, _ := range Cr {
			Cr[k] = 1.0/Cr[k]
		}
	} else  {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			Cr[k] = 1.0/Cr[k]
		}
	}
	return C
}

// Make a copy C of A and compute C += alpha for all elements in the matrix if list of indexes
// is empty. Otherwise compute C[indexes[i]] += alpha for indexes in column-major order.
func Add(A *FloatMatrix, alpha float64, indexes... int) *FloatMatrix {
	C := A.Copy()
	C.Add(alpha, indexes...)
	return C
}

// Make copy C of A and compute  C[indexes[i]] +=  values[i]. Indexes are in column-major order.
// Returns a new matrix.
func AddAt(A *FloatMatrix, values []float64, indexes []int) *FloatMatrix {
	C := A.Copy()
	if len(indexes) == 0 {
		return C
	}
	Cr := C.FloatArray()
	N := A.NumElements()
	for i, k := range indexes {
		if i >= len(values) {
			return C
		}
		if k < 0 {
			k = N + k
		}
		Cr[k] += values[i]
	}
	return C
}


// Compute element wise division C[i,j] = A[i,j] / B[i,j]. Returns new matrix.
func Div(A, B *FloatMatrix) *FloatMatrix {
	if ! A.SizeMatch(B.Size()) {
		return nil
	}
	C := A.Copy()
	C.Div(B)
	return C
}

// Compute element-wise product C[i,j] = A[i,j] * B[i,j]. Returns new matrix.
func Mul(A, B *FloatMatrix) *FloatMatrix {
	if ! A.SizeMatch(B.Size()) {
		return nil
	}
	C := A.Copy()
	C.Mul(B)
	return C
}

// Compute element-wise sum A + B + C +... Returns a new matrix.
func Plus(matrices ...*FloatMatrix) *FloatMatrix {
	if len(matrices) <= 1 {
		if len(matrices) == 1 {
			return matrices[0]
		}
		return nil
	}
	A := matrices[0]
	for _, B := range matrices[1:] {
		if ! A.SizeMatch(B.Size()) {
			return nil
		}
	}
	C := A.Copy()
	for _, B := range matrices[1:] {
		C.Plus(B)
	}
	return C
}

// Compute element-wise difference A - B - C -... Returns a new matrix.
func Minus(matrices ...*FloatMatrix) *FloatMatrix {
	if len(matrices) <= 1 {
		if len(matrices) == 1 {
			return matrices[0]
		}
		return nil
	}
	A := matrices[0]
	for _, B := range matrices[1:] {
		if ! A.SizeMatch(B.Size()) {
			return nil
		}
	}
	C := A.Copy()
	for _, B := range matrices[1:] {
		C.Minus(B)
	}
	return C
}


// Compute matrix product C = A * B where A is m*p and B is p*n.
// Returns a new m*n matrix.
func Times(A, B *FloatMatrix) *FloatMatrix {
	if A.Cols() != B.Rows() {
		return nil
	}
	rows := A.Rows()
	cols := B.Cols()
	C := FloatZeros(rows, cols)
	Cr := C.FloatArray()
	arow := make([]float64, A.Cols())
	bcol := make([]float64, B.Rows())
	for i := 0; i < rows; i++ {
		arow = A.GetRowArray(i, arow)
		for j := 0; j < cols; j++ {
			bcol = B.GetColumnArray(j, bcol)
			for k, _ := range arow {
				Cr[j*rows+i] += arow[k]*bcol[k]
			}
		}
	}
	return C
}


// Makes a copy C of A and applies function fn to elements of then new copy C. If indexes is
// non empty then function is applied to all elements of C indexed by indexes array.
// Otherwise function is applied to all elements of A. New value of element in C is fn(A[i]).
// Returns a new matrix.
func Apply(A *FloatMatrix, fn func(float64)float64, indexes ...int) *FloatMatrix {
	if A == nil {
		return nil
	}
	C := A.Copy()
	return C.Apply(fn, indexes...)
}

// Makes a copy C of A and applies function fn to elements of then new copy C. If indexes is
// non empty then function is applied to all elements of C indexed by indexes array.
// Otherwise function is applied to all elements of A. New value of element in C is fn(A[i], x).
// Returns a new matrix.
func ApplyConst(A *FloatMatrix, x float64, fn func(float64,float64)float64, indexes ...int) *FloatMatrix {
	if A == nil {
		return nil
	}
	C := A.Copy()
	return C.ApplyConst(x, fn, indexes...)
}

// Makes a copy C of A and applies function fn to elements of the new copy C pointed
// by the contexts of indexes array. New value of element in C is fn(A[indexes[i]], values[i]).
// Returns new matrix.
func ApplyConstValues(A *FloatMatrix, values []float64, fn func(float64,float64)float64, indexes []int) *FloatMatrix {
	if A == nil {
		return A
	}
	C := A.Copy()
	return C.ApplyConstValues(values, fn, indexes)
}

// Find element-wise maximum. 
func Max(A *FloatMatrix, indexes... int) float64 {
	res := math.Inf(-1)
	Ar := A.FloatArray()
	if len(indexes) == 0 {
		for _, v := range Ar {
			res = math.Max(res, v)
		}
	} else {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			res = math.Max(res, Ar[k])
		}
	}
	return res
}

// Find element-wise minimum. 
func Min(A *FloatMatrix, indexes... int) float64 {
	res := math.Inf(+1)
	Ar := A.FloatArray()
	if len(indexes) == 0 {
		for _, v := range Ar {
			res = math.Min(res, v)
		}
	} else {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			res = math.Min(res, Ar[k])
		}
	}
	return res
}

// Return sum of elements
func Sum(A *FloatMatrix, indexes... int) float64 {
	res := 0.0
	Ar := A.FloatArray()
	if len(indexes) == 0 {
		for _, v := range Ar {
			res += v
		}
	} else {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			res += Ar[k]
		}
	}
	return res
}

// Compute element-wise C = Exp(A). Returns a new matrix.
func Exp(A *FloatMatrix, indexes ...int) *FloatMatrix {
	return Apply(A, math.Exp, indexes...)
}

// Compute element-wise C = Log(A). Returns a new matrix.
func Log(A *FloatMatrix, indexes ...int)  *FloatMatrix {
	return Apply(A, math.Log, indexes...)
}

// Return copy of A with each element as Log10(A[i,j]).
func Log10(A *FloatMatrix, indexes ...int) *FloatMatrix {
	return Apply(A, math.Log10, indexes...)
}

// Return copy of A with each element as Log1p(A[i,j]). Returns nil for
// complex valued matrix.
func Log1p(A *FloatMatrix, indexes ...int) *FloatMatrix {
	return Apply(A, math.Log1p, indexes...)
}

// Return copy of A with each element as Log2(A[i,j]). Returns nil for
// complex valued matrix.
func Log2(A *FloatMatrix, indexes ...int) *FloatMatrix {
	return Apply(A, math.Log2, indexes...)
}

// Compute element-wise C = Pow(A). Returns a new matrix.
func Pow(A *FloatMatrix, exp float64, indexes ...int) *FloatMatrix {
	return ApplyConst(A, exp, math.Pow, indexes...)
}

// Compute element-wise C = Sqrt(A). Returns a new matrix.
func Sqrt(A *FloatMatrix, indexes ...int) *FloatMatrix {
	return Apply(A, math.Sqrt, indexes...)
}

// Compute element-wise C = Abs(A). Returns a new matrix.
func Abs(A *FloatMatrix, indexes ...int) *FloatMatrix {
	return Apply(A, math.Abs, indexes...)
}

// Local Variables:
// tab-width: 4
// End:
