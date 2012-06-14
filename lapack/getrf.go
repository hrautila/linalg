
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package lapack

import (
	"linalg"
	"matrix"
	"errors"
)

/*
 LU factorization of a general real or complex m by n matrix.

 Getrf(A, ipiv, m=A.Rows, n=A.Cols, ldA=max(1,A.Rows), offsetA=0)

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
func Getrf(A matrix.Matrix, ipiv []int32, opts ...linalg.Opt) error {
	ind := linalg.GetIndexOpts(opts...)
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
		ind.LDa = max(1, A.Rows())
	}
	if ind.LDa < max(1, ind.M) {
		return errors.New("lda")
	}
	if ind.OffsetA < 0 {
		return errors.New("offsetA")
	}
	sizeA := A.NumElements()
	if sizeA < ind.OffsetA+(ind.N-1)*ind.LDa+ind.M {
		return errors.New("sizeA")
	}
	if ipiv != nil && len(ipiv) < min(ind.N, ind.M) {
		return errors.New("size ipiv")
	}
	info := -1
	switch A.(type) {
	case *matrix.FloatMatrix:
		Aa := A.FloatArray()
		info = dgetrf(ind.M, ind.N, Aa[ind.OffsetA:], ind.LDa, ipiv)
	case *matrix.ComplexMatrix:
	}
	if info != 0 {
		return errors.New("Getrf call error")
	}
	return nil
}


// Local Variables:
// tab-width: 4
// End:
