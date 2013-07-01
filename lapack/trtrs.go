// Copyright (c) Harri Rautila, 2012,2013

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
 Solution of a triangular set of equations with multiple righthand
 sides.

 PURPOSE
 Solves set of equations
  A*X = B,   if trans is PNoTrans 
  A^T*X = B, if trans is PTrans
  A^H*X = B, if trans is PConjTrans

 B is n by nrhs and A is triangular of order n.

 ARGUMENTS
  A         float or complex matrix
  B         float or complex matrix.  Must have the same type as A.

 OPTIONS
  uplo      PLower or PUpper
  trans     PNoTrans, PTrans, PConjTrans
  diag      PNonUnit, PUnit
  n         nonnegative integer.  If negative, the default value is used.
  nrhs      nonnegative integer.  If negative, the default value is used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default
            value is used.
  ldB       positive integer.  ldB >= max(1,n).  If zero, the default
            value is used.
  offsetA   nonnegative integer
  offsetB   nonnegative integer;

*/
func Trtrs(A, B matrix.Matrix, opts ...linalg.Option) error {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return err
	}
	ind := linalg.GetIndexOpts(opts...)
	arows := ind.LDa
	brows := ind.LDb
	if ind.N < 0 {
		ind.N = A.Rows()
		if ind.N != A.Cols() {
			return onError("Trtrs: A not square")
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
		return onError("Trtrs: ldA")
	}
	if ind.LDb == 0 {
		ind.LDb = max(1, B.LeadingIndex())
		brows = max(1, B.Rows())
	}
	if ind.LDb < max(1, ind.N) {
		return onError("Trtrs: ldB")
	}
	if ind.OffsetA < 0 {
		return onError("Trtrs: offsetA")
	}
	sizeA := A.NumElements()
	if sizeA < ind.OffsetA+(ind.N-1)*arows+ind.N {
		return onError("Trtrs: sizeA")
	}
	if ind.OffsetB < 0 {
		return onError("Trtrs: offsetB")
	}
	sizeB := B.NumElements()
	if sizeB < ind.OffsetB+(ind.Nrhs-1)*brows+ind.N {
		return onError("Trtrs: sizeB")
	}
	if !matrix.EqualTypes(A, B) {
		return onError("Trtrs: arguments not of same type")
	}
	info := -1
	uplo := linalg.ParamString(pars.Uplo)
	trans := linalg.ParamString(pars.Trans)
	diag := linalg.ParamString(pars.Diag)
	switch A.(type) {
	case *matrix.FloatMatrix:
		Aa := A.(*matrix.FloatMatrix).FloatArray()
		Ba := B.(*matrix.FloatMatrix).FloatArray()
		info = dtrtrs(uplo, trans, diag, ind.N, ind.Nrhs, Aa[ind.OffsetA:], ind.LDa,
			Ba[ind.OffsetB:], ind.LDb)
	case *matrix.ComplexMatrix:
		return onError("Trtrs: complex not yet implmented")
	}
	if info != 0 {
		return onError(fmt.Sprintf("Trtrs lapack error: %d", info))
	}
	return nil
}

// Local Variables:
// tab-width: 4
// End:
