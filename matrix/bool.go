
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix


// Test if matrices A and B are equal. Returns false if sizes of the matrices
// do not match or if any A[i,j] != B[i,j]
func Equal(A, B Matrix) bool {
	if ! A.SizeMatch(B.Size()) {
		return false
	}
	rEQ := func(a, b float64) bool {
		return a == b
	}
	cEQ := func(c, d complex128) bool {
		return c == d
	}
	return applyTest(A, B, rEQ, cEQ)
}

// Test if matrices A and B are not equal. Returns false if sizes of the matrices
// do not match or if all A[i,j] == B[i,j]
func NotEqual(A, B Matrix) bool {
	if ! A.SizeMatch(B.Size()) {
		return false
	}
	return ! Equal(A, B)
}

// Test if parameter matrices have same element type, float or complex.
// It is based on method IsComplex(). Returns true if call IsComplex for
// all matrices return same result. Also returns true if parameter 
// count is <= 1.
func EqualElementTypes(As ...Matrix) bool {
	if len(As) <= 1 {
		return true
	}
	cmplx := As[0].IsComplex()
	for _, e := range As[1:] {
		if cmplx != e.IsComplex() {
			return false
		}
	}
	return true
}

// Test if matrices are of same type. Return true if all are same type
// as the first element, otherwise return false. For parameter count <=1
// return true.
func EqualTypes(As ...Matrix) bool {
	if len(As) <= 1 {
		return true
	}
	return As[0].EqualTypes(As...) 
}

// Apply element wise test between elements in A and B. Returns false
// if any test fails and conversely true if test succeeds for all elements
// of A and B.
func applyTest(A, B Matrix, rOp func(float64,float64)bool, cOp func(complex128,complex128)bool) bool {
	if A.IsComplex() && B.IsComplex() && cOp != nil {
		a := A.ComplexArray()
		b := B.ComplexArray()
		for k, _ := range a {
			if ! cOp(a[k], b[k]) {
				return false
			}
		}
		return true
	} else if ! A.IsComplex() && ! B.IsComplex() && rOp != nil {
		a := A.FloatArray()
		b := B.FloatArray()
		for k, _ := range a {
			if ! rOp(a[k], b[k]) {
				return false
			}
		}
		return true
	}
	return false
}


// Local Variables:
// tab-width: 4
// End:
