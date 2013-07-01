// Copyright (c) Harri Rautila, 2012, 2013

// This file is part of github.com/hrautila/linalg/lapack package.
// It is free software, distributed under the terms of GNU Lesser General Public 
// License Version 3, or any later version. See the COPYING tile included in this archive.

package lapack

import (
	//"errors"
	"fmt"
	"github.com/hrautila/linalg"
	"github.com/hrautila/matrix"
)

/*
 Solves a general real or complex set of linear equations.

 PURPOSE

 Solves A*X=B with A n by n real or complex.

 If ipiv is provided, then on exit A is overwritten with the details
 of the LU factorization, and ipiv contains the permutation matrix.
 If ipiv is not provided, then gesv() does not return the 
 factorization and does not modify A.  On exit B is replaced with
 the solution X.

 ARGUMENTS.
  A         float or complex matrix
  B         float or complex matrix.  Must have the same type as A.
  ipiv      int vector of length at least n

 OPTIONS:
  n         nonnegative integer.  If negative, the default value is used.
  nrhs      nonnegative integer.  If negative, the default value is used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default value is used.
  ldB       positive integer.  ldB >= max(1,n).  If zero, the default value is used.
  offsetA   nonnegative integer
  offsetA   nonnegative integer;
*/
func Gesv(A, B matrix.Matrix, ipiv []int32, opts ...linalg.Option) error {
	//pars, err := linalg.GetParameters(opts...)
	ind := linalg.GetIndexOpts(opts...)
	arows := ind.LDa
	brows := ind.LDb
	if ind.N < 0 {
		ind.N = A.Rows()
		if ind.N != A.Cols() {
			return onError("Gesv: A not square")
		}
	}
	if ind.Nrhs < 0 {
		ind.Nrhs = B.Cols()
	}
	if ind.N == 0 || ind.Nrhs == 0 {
		return nil
	}
	if ind.LDa == 0 {
		ind.LDa = max(1, A.LeadingIndex())
		arows = max(1, A.Rows())
	}
	if ind.LDa < max(1, ind.N) {
		return onError("Gesv: ldA")
	}
	if ind.LDb == 0 {
		ind.LDb = max(1, B.LeadingIndex())
		brows = max(1, B.Rows())
	}
	if ind.LDb < max(1, ind.N) {
		return onError("Gesv: ldB")
	}
	if ind.OffsetA < 0 {
		return onError("Gesv: offsetA")
	}
	if ind.OffsetB < 0 {
		return onError("Gesv: offsetB")
	}
	sizeA := A.NumElements()
	if sizeA < ind.OffsetA+(ind.N-1)*arows+ind.N {
		return onError("Gesv: sizeA")
	}
	sizeB := B.NumElements()
	if sizeB < ind.OffsetB+(ind.Nrhs-1)*brows+ind.N {
		return onError("Gesv: sizeB")
	}
	if ipiv != nil && len(ipiv) < ind.N {
		return onError("Gesv: size ipiv")
	}
	if !matrix.EqualTypes(A, B) {
		return onError("Gesv: arguments not of same type")
	}
	info := -1
	switch A.(type) {
	case *matrix.FloatMatrix:
		Aa := A.(*matrix.FloatMatrix).FloatArray()
		Ba := B.(*matrix.FloatMatrix).FloatArray()
		info = dgesv(ind.N, ind.Nrhs, Aa[ind.OffsetA:], ind.LDa, ipiv,
			Ba[ind.OffsetB:], ind.LDb)
	case *matrix.ComplexMatrix:
		return onError("Gesv: complex not yet implemented")
	}
	if info != 0 {
		return onError(fmt.Sprintf("Gesv: lapack error: %d", info))
	}
	return nil
}

// Local Variables:
// tab-width: 4
// End:
