
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import "math/cmplx"

// Compute in-place product A[i,j] *= alpha
func (A *ComplexMatrix) Scale(alpha complex128, indexes ...int) *ComplexMatrix {
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
func (A *ComplexMatrix) ScaleIndexes(indexes []int, values []complex128) *ComplexMatrix {
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


// Compute in-place sum A[i,j] += alpha
func (A *ComplexMatrix) Add(alpha complex128, indexes ...int) *ComplexMatrix {
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
func (A *ComplexMatrix) AddIndexes(indexes []int, values []complex128) *ComplexMatrix {
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


// Compute in-place negation -A[i,j]
func (A *ComplexMatrix) Neg() *ComplexMatrix {
	for k, v := range A.elements {
		A.elements[k] = -v
	}
	return A
}

// Compute element-wise division A /= B. Return A. If A and B sizes
// do not match A is returned unaltered.
func (A *ComplexMatrix) Div(B *ComplexMatrix) *ComplexMatrix {
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
func (A *ComplexMatrix) Mul(B *ComplexMatrix) *ComplexMatrix {
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
func (A *ComplexMatrix) Plus(B *ComplexMatrix) *ComplexMatrix {
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
func (A *ComplexMatrix) Minus(B *ComplexMatrix) *ComplexMatrix {
	if ! A.SizeMatch(B.Size()) {
		return A
	}
	for k, v := range B.elements {
		A.elements[k] -= v
	}
	return A
}



// Compute A = fn(A) by applying function fn element wise to A.
// If indexes array is non-empty function is applied to elements of A
// indexed by the contents of indexes.
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
// If indexes array is non-empty function is applied to elements of A
// indexed by the contents of indexes.
func (A *ComplexMatrix) ApplyConst(x complex128, fn func(complex128,complex128)complex128, indexes ...int) *ComplexMatrix {
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

// Compute A = fn(A, x) by applying function fn element wise to A.
//  For all i in indexes: A[indexes[i]] = fn(A[indexes[i]], values[i])
func (A *ComplexMatrix) ApplyConstValues(values []complex128, fn func(complex128,complex128)complex128, indexes []int) *ComplexMatrix {
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

// Compute in-place conjugate A[i,j]
func (A *ComplexMatrix) Conj() *ComplexMatrix {
	return A.Apply(cmplx.Conj)
}


// Compute in-place Exp(A)
func (A *ComplexMatrix) Exp() *ComplexMatrix {
	return A.Apply(cmplx.Exp)
}

// Compute in-place Log(A)
func (A *ComplexMatrix) Log() *ComplexMatrix {
	return A.Apply(cmplx.Log)
}

// Compute in-place Log10(A)
func (A *ComplexMatrix) Log10() *ComplexMatrix {
	return A.Apply(cmplx.Log10)
}

// Compute in-place Sqrt(A)
func (A *ComplexMatrix) Sqrt() *ComplexMatrix {
	return A.Apply(cmplx.Sqrt)
}

// Compute in-place Pow(A, x)
func (A *ComplexMatrix) Pow(x complex128) *ComplexMatrix {
	return A.ApplyConst(x, cmplx.Pow)
}


// Local Variables:
// tab-width: 4
// End:
