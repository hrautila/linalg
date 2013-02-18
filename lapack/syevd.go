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
 Eigenvalue decomposition of a real symmetric matrix
 (divide-and-conquer driver).

 PURPOSE

 Returns  eigenvalues/vectors of a real symmetric nxn matrix A.
 On exit, W contains the eigenvalues in ascending order.
 If jobz is PJobV, the (normalized) eigenvectors are also computed
 and returned in A.  If jobz is PJobNo, only the eigenvalues are
 computed, and the content of A is destroyed.

 ARGUMENTS
  A         float matrix
  W         float matrix of length at least n.  On exit, contains
            the computed eigenvalues in ascending order.

 OPTIONS
  jobz      PJobNo or PJobV
  uplo      PLower or PUpper
  n         integer.  If negative, the default value is used.
  ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the
            default value is used.
  offsetA   nonnegative integer
  offsetB   nonnegative integer;
*/
func Syevd(A, W matrix.Matrix, opts ...linalg.Option) error {
    if !matrix.EqualTypes(A, W) {
        return errors.New("Syevd: arguments not of same type")
    }
    switch A.(type) {
    case *matrix.FloatMatrix:
        Am := A.(*matrix.FloatMatrix)
        Wm := W.(*matrix.FloatMatrix)
        return SyevdFloat(Am, Wm, opts...)
    case *matrix.ComplexMatrix:
        return errors.New("Syevd: not a complex function")
    }
    return errors.New("Syevd: unknown types")
}

func SyevdFloat(A, W *matrix.FloatMatrix, opts ...linalg.Option) error {
    pars, err := linalg.GetParameters(opts...)
    if err != nil {
        return err
    }
    ind := linalg.GetIndexOpts(opts...)
    err = checkSyevd(ind, A, W)
    if err != nil {
        return err
    }
    if ind.N == 0 {
        return nil
    }
    jobz := linalg.ParamString(pars.Jobz)
    uplo := linalg.ParamString(pars.Uplo)
    Aa := A.FloatArray()
    Wa := W.FloatArray()
    info := dsyevd(jobz, uplo, ind.N, Aa[ind.OffsetA:], ind.LDa, Wa[ind.OffsetW:])
    if info != 0 {
        return errors.New(fmt.Sprintf("Syevd: lapack error %d", info))
    }
    return nil
}

func checkSyevd(ind *linalg.IndexOpts, A, W matrix.Matrix) error {
    if ind.N < 0 {
        ind.N = A.Rows()
        if ind.N != A.Cols() {
            return errors.New("Syevd: A not square")
        }
    }
    if ind.N == 0 {
        return nil
    }
    if ind.LDa == 0 {
        ind.LDa = max(1, A.LeadingIndex())
    }
    if ind.LDa < max(1, ind.N) {
        return errors.New("Syevd: lda")
    }
    if ind.OffsetA < 0 {
        return errors.New("Syevd: offsetA")
    }
    sizeA := A.NumElements()
	arows := max(1, A.Rows())
    if sizeA < ind.OffsetA+(ind.N-1)*arows+ind.N {
        return errors.New("Syevd: sizeA")
    }
    if ind.OffsetW < 0 {
        return errors.New("Syevd: offsetW")
    }
    sizeW := W.NumElements()
    if sizeW < ind.OffsetW+ind.N {
        return errors.New("Syevd: sizeW")
    }
    return nil
}

// Local Variables:
// tab-width: 4
// End:
