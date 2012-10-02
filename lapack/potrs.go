// Copyright (c) Harri Rautila, 2012

// This file is part of github.com/hrautila/linalg/lapack package.
// It is free software, distributed under the terms of GNU Lesser General Public 
// License Version 3, or any later version. See the COPYING tile included in this archive.

package lapack

import (
    "errors"
    "fmt"
    "github.com/hrautila/linalg"
    "github.com/hrautila/matrix"
)

/*
 Solves a real symmetric or complex Hermitian positive definite set
 of linear equations, given the Cholesky factorization computed by
 potrf() or posv().

 PURPOSE

 Solves
   A*X = B

 where A is n by n, real symmetric or complex Hermitian and positive definite,
 and B is n by nrhs. On entry, A contains the Cholesky factor, as
 returned by Posv() or Potrf().  On exit B is replaced by the solution X.

 ARGUMENTS
  A         float or complex matrix
  B         float or complex matrix.  Must have the same type as A.

 OPTIONS
  uplo      PLower or PUpper
  n         nonnegative integer.  If negative, the default value is used.
  nrhs      nonnegative integer.  If negative, the default value is used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default
            value is used.
  ldB       positive integer.  ldB >= max(1,n).  If zero, the default
            value is used.
  offsetA   nonnegative integer
  offsetB   nonnegative integer;

*/
func Potrs(A, B matrix.Matrix, opts ...linalg.Option) error {
    pars, err := linalg.GetParameters(opts...)
    if err != nil {
        return err
    }
    ind := linalg.GetIndexOpts(opts...)
    if ind.N < 0 {
        ind.N = A.Rows()
    }
    if ind.Nrhs < 0 {
        ind.Nrhs = B.Cols()
    }
    if ind.N == 0 || ind.Nrhs == 0 {
        return nil
    }
    if ind.LDa == 0 {
        ind.LDa = max(1, A.Rows())
    }
    if ind.LDa < max(1, ind.N) {
        return errors.New("Potrs: ldA")
    }
    if ind.LDb == 0 {
        ind.LDb = max(1, B.Rows())
    }
    if ind.LDb < max(1, ind.N) {
        return errors.New("Potrs: ldB")
    }
    if ind.OffsetA < 0 {
        return errors.New("Potrs: offsetA")
    }
    if A.NumElements() < ind.OffsetA+(ind.N-1)*ind.LDa+ind.N {
        return errors.New("Potrs: sizeA")
    }
    if ind.OffsetB < 0 {
        return errors.New("Potrs: offsetB")
    }
    if B.NumElements() < ind.OffsetB+(ind.Nrhs-1)*ind.LDb+ind.N {
        return errors.New("Potrs: sizeB")
    }
    if !matrix.EqualTypes(A, B) {
        return errors.New("Potrs: arguments not of same types")
    }
    info := -1
    switch A.(type) {
    case *matrix.FloatMatrix:
        Aa := A.(*matrix.FloatMatrix).FloatArray()
        Ba := B.(*matrix.FloatMatrix).FloatArray()
        uplo := linalg.ParamString(pars.Uplo)
        info = dpotrs(uplo, ind.N, ind.Nrhs, Aa[ind.OffsetA:], ind.LDa,
            Ba[ind.OffsetB:], ind.LDb)
    case *matrix.ComplexMatrix:
        return errors.New("Potrs: complex not implemented yet")
    }
    if info != 0 {
        return errors.New(fmt.Sprintf("Potrs: lapack error %d", info))
    }
    return nil
}

// Local Variables:
// tab-width: 4
// End:
