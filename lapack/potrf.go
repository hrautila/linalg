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
 Cholesky factorization of a real symmetric or complex Hermitian
 positive definite matrix.

 PURPOSE

 Factors A as A=L*L^T or A = L*L^H, where A is n by n, real
 symmetric or complex Hermitian, and positive definite.

 On exit, if uplo=PLower, the lower triangular part of A is replaced
 by L.  If uplo=PUpper, the upper triangular part is replaced by L^T
 or L^H.

 ARGUMENTS
  A         float or complex matrix

 OPTIONS
  uplo      PLower or PUpper
  n         nonnegative integer.  If negative, the default value is used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default
            value is used.
  offsetA   nonnegative integer

*/
func Potrf(A matrix.Matrix, opts ...linalg.Option) error {
    switch A.(type) {
    case *matrix.FloatMatrix:
        return PotrfFloat(A.(*matrix.FloatMatrix), opts...)
    case *matrix.ComplexMatrix:
        return onError("Potrf: complex not implemented yet")
    }
    return onError("Potrf unknown types")
}

func PotrfFloat(A *matrix.FloatMatrix, opts ...linalg.Option) error {
    pars, err := linalg.GetParameters(opts...)
    if err != nil {
        return err
    }
    ind := linalg.GetIndexOpts(opts...)
    err = checkPotrf(ind, A)
    if ind.N == 0 {
        return nil
    }
    Aa := A.FloatArray()
    uplo := linalg.ParamString(pars.Uplo)
    info := dpotrf(uplo, ind.N, Aa[ind.OffsetA:], ind.LDa)
    if info != 0 {
        return onError(fmt.Sprintf("Potrf: lapack error %d", info))
    }
    return nil
}

func checkPotrf(ind *linalg.IndexOpts, A matrix.Matrix) error {
    arows := ind.LDa
    if ind.N < 0 {
        ind.N = A.Rows()
        if ind.N != A.Cols() {
            return onError("Potrf: not square")
        }
    }
    if ind.N == 0 {
        return nil
    }
    if ind.LDa == 0 {
        ind.LDa = max(1, A.LeadingIndex())
        arows = max(1, A.Rows())
    }
    if ind.LDa < max(1, ind.N) {
        return onError("Potrf: lda")
    }
    if ind.OffsetA < 0 {
        return onError("Potrf: offsetA")
    }
    if A.NumElements() < ind.OffsetA+(ind.N-1)*arows+ind.N {
        return onError("Potrf: sizeA")
    }
    return nil
}

// Local Variables:
// tab-width: 4
// End:
