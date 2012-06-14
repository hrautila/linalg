
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

// Test for equality. Return true if for all i,j: all A[i,j] = B[i,j]
func (A *FloatMatrix) Equal(B *FloatMatrix) bool {
	if ! A.SizeMatch(B.Size()) {
		return false
	}
	for k, _ := range A.elements {
		if A.elements[k] != B.elements[k] {
			return false
		}
	}
	return true
}

// Test for element wise less-than. Return true if for all i,j: A[i,j] < B[i,j]
func (A *FloatMatrix) Less(B *FloatMatrix) bool {
	if ! A.SizeMatch(B.Size()) {
		return false
	}
	for k, _ := range A.elements {
		if A.elements[k] >= B.elements[k] {
			return false
		}
	}
	return true

}

// Test for element wise less-or-equal.
// Return true if for all i,j: A[i,j] <= B[i,j]
func (A *FloatMatrix) LessOrEqual(B *FloatMatrix) bool {
	if ! A.SizeMatch(B.Size()) {
		return false
	}
	for k, _ := range A.elements {
		if A.elements[k] > B.elements[k] {
			return false
		}
	}
	return true
}

// Test for element wise greater-than.
// Return true if for all i,j: A[i,j] > B[i,j]
func (A *FloatMatrix) Greater(B *FloatMatrix) bool {
	if ! A.SizeMatch(B.Size()) {
		return false
	}
	for k, _ := range A.elements {
		if A.elements[k] <= B.elements[k] {
			return false
		}
	}
	return true
}

// Test for element wise greater-than-or-equal.
// Return true if for all i,j: A[i,j] >= B[i,j]
func (A *FloatMatrix) GreaterOrEqual(B *FloatMatrix) bool {
	if ! A.SizeMatch(B.Size()) {
		return false
	}
	for k, _ := range A.elements {
		if A.elements[k] < B.elements[k] {
			return false
		}
	}
	return true
}


// Local Variables:
// tab-width: 4
// End:



	
