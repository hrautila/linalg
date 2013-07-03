// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/linalg/lapack package.
// It is free software, distributed under the terms of GNU Lesser General Public 
// License Version 3, or any later version. See the COPYING tile included in this archive.

package lapack

import (
	//"errors"
	"github.com/hrautila/linalg"
	"github.com/hrautila/matrix"
)

/*
 LU factorization of a general real or complex m by n matrix.

 PURPOSE

 On exit, A is replaced with L, U in the factorization P*A = L*U
 and ipiv contains the permutation:
 P = P_min{m,n} * ... * P2 * P1 where Pi interchanges rows i and
 ipiv[i] of A (using the Fortran convention, i.e., the first row
 is numbered 1).

 ARGUMENTS
  A         float or complex matrix
  ipiv      int vector of length at least min(m,n)

 OPTIONS
  m         nonnegative integer.  If negative, the default value is used.
  n         nonnegative integer.  If negative, the default value is used.
  ldA       positive integer.  ldA >= max(1,m).  If zero, the default
            value is used.
  offsetA   nonnegative integer

*/
func Getrf(A matrix.Matrix, ipiv []int32, opts ...linalg.Option) error {
	ind := linalg.GetIndexOpts(opts...)
	arows := ind.LDa
	if ind.M < 0 {
		ind.M = A.Rows()
	}
	if ind.N < 0 {
		ind.N = A.Cols()
	}
	if ind.N == 0 || ind.M == 0 {
		return nil
	}
	if ind.LDa == 0 {
		ind.LDa = max(1, A.LeadingIndex())
		arows = max(1, A.Rows())
	}
	if ind.LDa < max(1, ind.M) {
		return onError("lda")
	}
	if ind.OffsetA < 0 {
		return onError("offsetA")
	}
	sizeA := A.NumElements()
	if sizeA < ind.OffsetA+(ind.N-1)*arows+ind.M {
		return onError("sizeA")
	}
	if ipiv != nil && len(ipiv) < min(ind.N, ind.M) {
		return onError("size ipiv")
	}
	info := -1
	switch A.(type) {
	case *matrix.FloatMatrix:
		Aa := A.(*matrix.FloatMatrix).FloatArray()
		info = dgetrf(ind.M, ind.N, Aa[ind.OffsetA:], ind.LDa, ipiv)
	case *matrix.ComplexMatrix:
	}
	if info != 0 {
		return onError("Getrf call error")
	}
	return nil
}

// Local Variables:
// tab-width: 4
// End:
