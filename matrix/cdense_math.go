
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import "math/cmplx"

// Compute in-place product A[i,j] *= alpha
func (A *ComplexMatrix) Scale(alpha complex128) *ComplexMatrix {
	for k, v := range A.elements {
		A.elements[k] = alpha * v
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


// Compute in-place negation -A[i,j]
func (A *ComplexMatrix) Neg() *ComplexMatrix {
	for k, v := range A.elements {
		A.elements[k] = -v
	}
	return A
}

// Compute in-place conjugate A[i,j]
func (A *ComplexMatrix) Conj() *ComplexMatrix {
	for k, v := range A.elements {
		A.elements[k] = cmplx.Conj(v)
	}
	return A
}


// Compute A = fn(C) by applying function fn element wise to A. For all i, j:
// A[i,j] = fn(C[i,j]). If C == nil reduces to A[i,j] = fn(A[i,j]). Returns self. 
func (A *ComplexMatrix) Apply(fn func(complex128)complex128, indexes ...int) *ComplexMatrix {
	if len(indexes) == 0 {
		for k, v := range A.elements {
			A.elements[k] = fn(v)
		}
	} else {
		N := A.NumElements()
		for k, v := range A.elements {
			if k < 0 {
				k = k + N
			}
			A.elements[k] = fn(v)
		}
	}
	return A
}

// Compute A = fn(A, x) by applying function fn element wise to A.
func (A *ComplexMatrix) Apply2(fn func(complex128,complex128)complex128, x complex128, indexes ...int) *ComplexMatrix {
	if len(indexes) == 0 {
		for k, v := range A.elements {
			A.elements[k] = fn(v, x)
		}
	} else {
		N := A.NumElements()
		for k, v := range A.elements {
			if k < 0 {
				k = k + N
			}
			A.elements[k] = fn(v, x)
		}
	}
	return A
}


// Local Variables:
// tab-width: 4
// End:
