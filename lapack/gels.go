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

 Solves A*X=B with A m by n real or complex.

 ARGUMENTS.
  A         float or complex matrix
  B         float or complex matrix.  Must have the same type as A.

 OPTIONS:
  trans     
  m         nonnegative integer.  If negative, the default value is used.
  n         nonnegative integer.  If negative, the default value is used.
  nrhs      nonnegative integer.  If negative, the default value is used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default value is used.
  ldB       positive integer.  ldB >= max(1,n).  If zero, the default value is used.
*/
func Gels(A, B matrix.Matrix, opts ...linalg.Option) error {
    pars, _ := linalg.GetParameters(opts...)
    ind := linalg.GetIndexOpts(opts...)
    arows := ind.LDa
    brows := ind.LDb
    if ind.M < 0 {
        ind.M = A.Rows()
    }
    if ind.N < 0 {
        ind.N = A.Cols()
    }
    if ind.Nrhs < 0 {
        ind.Nrhs = B.Cols()
    }
    if ind.M == 0 || ind.N == 0 || ind.Nrhs == 0 {
        return nil
    }
    if ind.LDa == 0 {
        ind.LDa = max(1, A.LeadingIndex())
        arows = max(1, A.Rows())
    }
    if ind.LDa < max(1, ind.M) {
        return onError("Gesv: ldA")
    }
    if ind.LDb == 0 {
        ind.LDb = max(1, B.LeadingIndex())
        brows = max(1, B.Rows())
    }
    if ind.LDb < max(ind.M, ind.N) {
        return onError("Gesv: ldB")
    }
    if !matrix.EqualTypes(A, B) {
        return onError("Gesv: arguments not of same type")
    }
	_, _ = arows, brows // todo!! something
    info := -1
    trans := linalg.ParamString(pars.Trans)
    switch A.(type) {
    case *matrix.FloatMatrix:
        Aa := A.(*matrix.FloatMatrix).FloatArray()
        Ba := B.(*matrix.FloatMatrix).FloatArray()
        info = dgels(trans, ind.M, ind.N, ind.Nrhs, Aa[ind.OffsetA:], ind.LDa,
            Ba[ind.OffsetB:], ind.LDb)
    case *matrix.ComplexMatrix:
        return onError("Gels: complex not yet implemented")
    }
    if info != 0 {
        return onError(fmt.Sprintf("Gels: lapack error: %d", info))
    }
    return nil
}

// Local Variables:
// tab-width: 4
// End:
