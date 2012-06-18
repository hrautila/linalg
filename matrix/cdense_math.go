
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

// Compute in-place product A[i,j] *= alpha
func (A *ComplexMatrix) Mult(alpha complex128) *ComplexMatrix {
	for k, v := range A.elements {
		A.elements[k] = alpha * v
	}
	return A
}

// Compute in-place division A[i,j] /= alpha
func (A *ComplexMatrix) Div(alpha complex128) *ComplexMatrix {
	for k, v := range A.elements {
		A.elements[k] = v / alpha
	}
	return A
}

// Compute in-place sum A[i,j] += alpha
func (A *ComplexMatrix) Add(alpha complex128) *ComplexMatrix {
	for k, v := range A.elements {
		A.elements[k] = v + alpha 
	}
	return A
}

// Compute in-place difference A[i,j] -= alpha
func (A *ComplexMatrix) Sub(alpha complex128) *ComplexMatrix {
	for k, v := range A.elements {
		A.elements[k] = v - alpha
	}
	return A
}

// Compute in-place negation -A[i,j]
func (A *ComplexMatrix) Neg() *ComplexMatrix {
	for k, v := range A.elements {
		A.elements[k] = -v
	}
	return A
}

// Compute sum C = A + B. Returns a new matrix.
func (A *ComplexMatrix) Plus(B *ComplexMatrix) *ComplexMatrix {
	if A.Rows() != B.Rows() || A.Cols() != B.Cols() {
		return nil
	}
	C := ComplexZeros(A.Rows(), A.Cols())
	for k, _ := range A.elements {
		C.elements[k] = A.elements[k] + B.elements[k]
	}
	return C
}

// Compute difference C = A - B. Returns a new matrix.
func (A *ComplexMatrix) Minus(B *ComplexMatrix) *ComplexMatrix {
	if A.Rows() != B.Rows() || A.Cols() != B.Cols() {
		return nil
	}
	C := ComplexZeros(A.Rows(), A.Cols())
	for k, _ := range A.elements {
		C.elements[k] = A.elements[k] - B.elements[k]
	}
	return C
}

// Compute product C = A * B. Returns a new matrix.
func (A *ComplexMatrix) Times(B *ComplexMatrix) *ComplexMatrix {
	if A.Cols() != B.Rows() {
		return nil
	}
	C := ComplexZeros(A.Rows(), B.Cols())
	arow := make([]complex128, A.Cols())
	bcol := make([]complex128, B.Rows())
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


// Compute C = fn(A) by applying function fn element wise to A. For all i, j:
// C[i,j] = fn(A[i,j]). Returns new matrix.
func (A *ComplexMatrix) Apply(fn func(complex128)complex128) *ComplexMatrix {
	C := ComplexZeros(A.Rows(), A.Cols())
	for k,v := range A.elements {
		C.elements[k] = fn(v)
	}
	return C
}

// Compute C = fn(A, x) by applying function fn element wise to A. For all i, j:
// C[i,j] = fn(A[i,j], x). Returns new matrix.
func (A *ComplexMatrix) Apply2(fn func(complex128,complex128)complex128, x complex128) *ComplexMatrix {
	C := ComplexZeros(A.Rows(), A.Cols())
	for k,v := range A.elements {
		C.elements[k] = fn(v, x)
	}
	return C
}

// Compute C = fn(A) by applying function fn to all elements in indexes.
// For all i in indexes: C[i] = fn(A[i]).
// If C is nil then computes inplace A = fn(A). If C is not nil then sizes of A and C must match.
// Returns pointer to the result matrix.
func (A *ComplexMatrix) ApplyToIndexes(C *ComplexMatrix, indexes []int, fn func(complex128)complex128) *ComplexMatrix {
	if C != nil && ! A.SizeMatch(C.Size()) {
		return nil
	}
	B := C
	if C == nil {
		B = A
	}
	for _,v := range indexes {
		B.elements[v] = fn(A.elements[v])
	}
	return B
}


// Local Variables:
// tab-width: 4
// End:
