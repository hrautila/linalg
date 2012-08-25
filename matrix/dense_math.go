
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import "math"


// Compute in-place A *= alpha for all elements in the matrix if list of indexes
// is empty. Otherwise compute A[i] *= alpha for indexes in column-major order.
func (A *FloatMatrix) Scale(alpha float64, indexes... int) *FloatMatrix {
	if len(indexes) == 0 {
		for k, _ := range A.elements {
			A.elements[k] *= alpha
		}
	} else  {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			A.elements[k] *= alpha
		}
	}
	return A
}

// Compute in-place A[indexes[i]] *= values[i]. Indexes are in column-major order.
func (A *FloatMatrix) ScaleIndexes(indexes []int, values []float64) *FloatMatrix {
	if len(indexes) == 0 {
		return A
	}
	N := A.NumElements()
	for i, k := range indexes {
		if i >= len(values) {
			return A
		}
		if k < 0 {
			k = N + k
		}
		A.elements[k] *= values[i]
	}
	return A
}


// Compute in-place remainder A[i,j] %= alpha
func (A *FloatMatrix) Mod(alpha float64, indexes... int) *FloatMatrix {
	if len(indexes) == 0 {
		for k, _ := range A.elements {
			A.elements[k] = math.Mod(A.elements[k], alpha)
		}
	} else  {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			A.elements[k] = math.Mod(A.elements[k], alpha)
		}
	}
	return A
}

// Compute in-place inverse A[i,j] = 1.0/A[i,j]. If indexes is empty calculates for
// all elements
func (A *FloatMatrix) Inv(indexes... int) *FloatMatrix {
	if len(indexes) == 0 {
		for k, _ := range A.elements {
			A.elements[k] = 1.0/A.elements[k]
		}
	} else  {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			A.elements[k] = 1.0/A.elements[k]
		}
	}
	return A
}

// Compute in-place A += alpha for all elements in the matrix if list of indexes
// is empty. Otherwise compute A[i] += alpha for indexes in column-major order.
func (A *FloatMatrix) Add(alpha float64, indexes... int) *FloatMatrix {
	if len(indexes) == 0 {
		for k, _ := range A.elements {
			A.elements[k] += alpha
		}
	} else  {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			A.elements[k] += alpha
		}
	}
	return A
}

// Compute in-place A[indexes[i]] += values[i]. Indexes are in column-major order.
func (A *FloatMatrix) AddIndexes(indexes []int, values []float64) *FloatMatrix {
	if len(indexes) == 0 {
		return A
	}
	N := A.NumElements()
	for i, k := range indexes {
		if i >= len(values) {
			return A
		}
		if k < 0 {
			k = N + k
		}
		A.elements[k] += values[i]
	}
	return A
}


// Compute element-wise division A /= B. Return A. If A and B sizes
// do not match A is returned unaltered.
func (A *FloatMatrix) Div(B *FloatMatrix) *FloatMatrix {
	if ! A.SizeMatch(B.Size()) {
		return A
	}
	for k, v := range B.elements {
		A.elements[k] /= v
	}
	return A
}

// Compute element-wise product A *= B. Return A. If A and B sizes
// do not match A is returned unaltered.
func (A *FloatMatrix) Mul(B *FloatMatrix) *FloatMatrix {
	if ! A.SizeMatch(B.Size()) {
		return A
	}
	for k, v := range B.elements {
		A.elements[k] *= v
	}
	return A
}

// Compute element-wise sum A += B. Return A. If A and B sizes
// do not match A is returned unaltered.
func (A *FloatMatrix) Plus(B *FloatMatrix) *FloatMatrix {
	if ! A.SizeMatch(B.Size()) {
		return A
	}
	for k, v := range B.elements {
		A.elements[k] += v
	}
	return A
}

// Compute element-wise difference A -= B. Return A. If A and B sizes
// do not match A is returned unaltered.
func (A *FloatMatrix) Minus(B *FloatMatrix) *FloatMatrix {
	if ! A.SizeMatch(B.Size()) {
		return A
	}
	for k, v := range B.elements {
		A.elements[k] -= v
	}
	return A
}


// Compute matrix product C = A * B where A is m*p and B is p*n.
// Returns a new m*n matrix.
func (A *FloatMatrix) Times(B *FloatMatrix) *FloatMatrix {
	if A.Cols() != B.Rows() {
		return nil
	}
	rows := A.Rows()
	cols := B.Cols()
	C := FloatZeros(rows, cols)
	arow := make([]float64, A.Cols())
	bcol := make([]float64, B.Rows())
	for i := 0; i < rows; i++ {
		arow = A.GetRowArray(i, arow)
		for j := 0; j < cols; j++ {
			bcol = B.GetColumnArray(j, bcol)
			for k, _ := range arow {
				C.elements[j*rows+i] += arow[k]*bcol[k]
			}
		}
	}
	return C
}


// Compute A = fn(C) by applying function fn element wise to C.
// For all i, j: A[i,j] = fn(C[i,j]). If C is nil then computes inplace
// A = fn(A). If C is not nil then sizes of A and C must match.
// Returns pointer to self.
func (A *FloatMatrix) Apply(fn func(float64)float64, indexes ...int) *FloatMatrix {
	if len(indexes) == 0 {
		for k,v := range A.elements {
			A.elements[k] = fn(v)
		}
	} else {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k += N
			}
			A.elements[k] = fn(A.elements[k])
		}
	}
	return A
}

// Compute A = fn(C) by applying function fn element wise to C.
// For all i, j: A[i,j] = fn(C[i,j]). If C is nil then computes inplace
// A = fn(A). If C is not nil then sizes of A and C must match.
// Returns pointer to self.
func (A *FloatMatrix) ApplyConst(x float64, fn func(float64,float64)float64, indexes ...int) *FloatMatrix {
	if len(indexes) == 0 {
		for k,v := range A.elements {
			A.elements[k] = fn(v, x)
		}
	} else {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k += N
			}
			A.elements[k] = fn(A.elements[k], x)
		}
	}
	return A
}

// Compute A = fn(C) by applying function fn to all elements in indexes.
// For all i in indexes: A[i] = fn(C[i]).
// If C is nil then computes inplace A = fn(A). If C is not nil then sizes of A and C must match.
// Returns pointer to self.
func (A *FloatMatrix) ApplyToIndexes(C *FloatMatrix, indexes []int, fn func(float64)float64) *FloatMatrix {
	if C != nil && ! A.SizeMatch(C.Size()) {
		return nil
	}
	B := C
	if C == nil {
		B = A
	}
	if len(indexes) > 0 {
		for _,v := range indexes {
			A.elements[v] = fn(B.elements[v])
		}
	}
	return A
}

// Compute A = fn(C, x) by applying function fn element wise to C.
// For all i, j: A[i,j] = fn(C[i,j], x). 
func (A *FloatMatrix) ApplyConstValues(values []float64, fn func(float64,float64)float64, indexes []int) *FloatMatrix {
	N := A.NumElements()
	for i, k := range indexes {
		if i > len(values) {
			return A
		}
		if k < 0 {
			k += N
		}
		A.elements[k] = fn(A.elements[k], values[i])
	}
	return A
}

// Find element-wise maximum. 
func (A *FloatMatrix) Max(indexes... int) float64 {
	m := math.Inf(-1)
	if len(indexes) == 0 {
		for _, v := range A.elements {
			m = math.Max(m, v)
		}
	} else {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			m = math.Max(m, A.elements[k])
		}
	}
	return m
}

// Find element-wise minimum. 
func (A *FloatMatrix) Min(indexes... int) float64 {
	m := math.Inf(+1)
	if len(indexes) == 0 {
		for _, v := range A.elements {
			m = math.Min(m, v)
		}
	} else {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			m = math.Min(m, A.elements[k])
		}
	}
	return m
}

// Return sum of elements
func (A *FloatMatrix) Sum(indexes... int) float64 {
	m := 0.0
	if len(indexes) == 0 {
		for _, v := range A.elements {
			m += v
		}
	} else {
		N := A.NumElements()
		for _, k := range indexes {
			if k < 0 {
				k = N + k
			}
			m += A.elements[k]
		}
	}
	return m
}

// Compute element-wise A = Exp(A).
func (A *FloatMatrix) Exp() *FloatMatrix {
	return A.Apply(math.Exp)
}

// Compute element-wise A = Sqrt(A).
func (A *FloatMatrix) Sqrt() *FloatMatrix {
	return A.Apply(math.Sqrt)
}

// Compute element-wise A = Log(A).
func (A *FloatMatrix) Log() *FloatMatrix {
	return A.Apply(math.Log)
}
		
// Compute element-wise A = Pow(A).
func (A *FloatMatrix) Pow(exp float64) *FloatMatrix {
	return A.ApplyConst(exp, math.Pow)
}
		

// Local Variables:
// tab-width: 4
// End:
