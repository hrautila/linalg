
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import "math"

// Compute in-place product A[i,j] *= alpha
func (A *FloatMatrix) Mult(alpha float64) *FloatMatrix {
	for k, v := range A.elements {
		A.elements[k] = alpha * v
	}
	return A
}

// Compute in-place division A[i,j] /= alpha
func (A *FloatMatrix) Div(alpha float64) *FloatMatrix {
	for k, v := range A.elements {
		A.elements[k] = v / alpha
	}
	return A
}

// Compute in-place remainder A[i,j] %= alpha
func (A *FloatMatrix) Mod(alpha float64) *FloatMatrix {
	for k, v := range A.elements {
		A.elements[k] = math.Mod(v, alpha)
	}
	return A
}

// Compute in-place sum A[i,j] += alpha
func (A *FloatMatrix) Add(alpha float64) *FloatMatrix {
	for k, v := range A.elements {
		A.elements[k] = v + alpha 
	}
	return A
}

// Compute in-place difference A[i,j] -= alpha
func (A *FloatMatrix) Sub(alpha float64) *FloatMatrix {
	for k, v := range A.elements {
		A.elements[k] = v - alpha
	}
	return A
}

// Compute in-place negation -A[i,j]
func (A *FloatMatrix) Neg() *FloatMatrix {
	for k, v := range A.elements {
		A.elements[k] = -v
	}
	return A
}

// Compute sum C = A + B. Returns a new matrix.
func (A *FloatMatrix) Plus(B *FloatMatrix) *FloatMatrix {
	if ! A.SizeMatch(B.Size()) {
		return nil
	}
	C := FloatZeros(A.Rows(), A.Cols())
	for k, _ := range A.elements {
		C.elements[k] = A.elements[k] + B.elements[k]
	}
	return C
}

// Compute difference C = A - B. Returns a new matrix.
func (A *FloatMatrix) Minus(B *FloatMatrix) *FloatMatrix {
	if ! A.SizeMatch(B.Size()) {
		return nil
	}
	C := FloatZeros(A.Rows(), A.Cols())
	for k, _ := range A.elements {
		C.elements[k] = A.elements[k] - B.elements[k]
	}
	return C
}

// Compute product C = A * B. Returns a new matrix.
func (A *FloatMatrix) Times(B *FloatMatrix) *FloatMatrix {
	if A.Cols() != B.Rows() {
		return nil
	}
	C := FloatZeros(A.Rows(), B.Cols())
	arow := make([]float64, A.Cols())
	bcol := make([]float64, B.Rows())
	for i := 0; i < A.Rows(); i++ {
		for j := 0; j < B.Cols(); j++ {
			arow = A.GetRow(i, arow)
			bcol = B.GetColumn(j, bcol)
			for k, _ := range arow {
				C.elements[i*A.Rows()+j] += arow[k]*bcol[k]
			}
		}
	}
	return C
}


// Compute C = fn(A) by applying function fn element wise to A.
// For all i, j: C[i,j] = fn(A[i,j]). If C is nil then computes inplace
// A = fn(A). If C is not nil then sizes of A and C must match.
// Returns pointer to the result matrix.
func (A *FloatMatrix) Apply(C *FloatMatrix, fn func(float64)float64) *FloatMatrix {
	if C != nil && ! A.SizeMatch(C.Size()) {
		return nil
	}
	B := C
	if C == nil {
		B = A
	}
	for k,v := range A.elements {
		B.elements[k] = fn(v)
	}
	return B
}

// Compute A = fn(A, x) by applying function fn element wise to A.
// For all i, j: A[i,j] = fn(A[i,j], x). 
func (A *FloatMatrix) Apply2(C *FloatMatrix, fn func(float64,float64)float64, x float64) *FloatMatrix {

	if C != nil && ! A.SizeMatch(C.Size()) {
		return nil
	}
	B := C
	if C == nil {
		B = A
	}
	for k,v := range A.elements {
		B.elements[k] = fn(v, x)
	}
	return B
}

// Find element-wise maximum. 
func (A *FloatMatrix) Max() float64 {
	m := math.Inf(-1)
	for _, v := range A.elements {
		m = math.Max(m, v)
	}
	return m
}

// Find element-wise minimum. 
func (A *FloatMatrix) Min() float64 {
	m := math.Inf(+1)
	for _, v := range A.elements {
		m = math.Min(m, v)
	}
	return m
}

// Compute C = Exp(A). Returns a new matrix.
func (A *FloatMatrix) Exp() *FloatMatrix {
	C := FloatZeros(A.Rows(), A.Cols())
	return A.Apply(C, math.Exp)
}

// Compute C = Log(A). Returns a new matrix.
func (A *FloatMatrix) Log() *FloatMatrix {
	C := FloatZeros(A.Rows(), A.Cols())
	return A.Apply(C, math.Log)
}
		
// Local Variables:
// tab-width: 4
// End:
