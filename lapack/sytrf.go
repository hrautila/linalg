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
 LDL^T factorization of a real or complex symmetric matrix.

 PURPOSE
 Computes the LDL^T factorization of a real or complex symmetric
 n by n matrix  A.  On exit, A and ipiv contain the details of the
 factorization.

 ARGUMENTS
  A         float or complex matrix
  ipiv      int vector of length at least n

 OPTIONS
  uplo      PLower or PUpper
  n         nonnegative integer.  If negative, the default value is  used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default
            value is used.
  offsetA   nonnegative integer;

*/
func Sytrf(A matrix.Matrix, ipiv []int32, opts ...linalg.Option) error {
    switch A.(type) {
    case *matrix.FloatMatrix:
        return SytrfFloat(A.(*matrix.FloatMatrix), ipiv, opts...)
    case *matrix.ComplexMatrix:
        return SytrfComplex(A.(*matrix.ComplexMatrix), ipiv, opts...)
    }
    return errors.New("Sytrf: unknown types")
}

func SytrfFloat(A *matrix.FloatMatrix, ipiv []int32, opts ...linalg.Option) error {
    pars, err := linalg.GetParameters(opts...)
    if err != nil {
        return err
    }
    ind := linalg.GetIndexOpts(opts...)
    err = checkSytrf(ind, A, ipiv)
    if err != nil {
        return err
    }
    if ind.N == 0 {
        return nil
    }
    Aa := A.FloatArray()
    uplo := linalg.ParamString(pars.Uplo)
    info := dsytrf(uplo, ind.N, Aa[ind.OffsetA:], ind.LDa, ipiv)
    if info != 0 {
        return errors.New(fmt.Sprintf("Sytrf: lapack error %d", info))
    }
    return nil
}

func SytrfComplex(A *matrix.ComplexMatrix, ipiv []int32, opts ...linalg.Option) error {
    return errors.New(fmt.Sprintf("Sytrf: complex not yet implemented"))
}

func checkSytrf(ind *linalg.IndexOpts, A matrix.Matrix, ipiv []int32) error {
	arows := ind.LDa
    if ind.N < 0 {
        ind.N = A.Rows()
        if ind.N != A.Cols() {
            return errors.New("A not square")
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
        return errors.New("Sytrf: lda")
    }
    if ind.OffsetA < 0 {
        return errors.New("Sytrf: offsetA")
    }
    sizeA := A.NumElements()
    if sizeA < ind.OffsetA+(ind.N-1)*arows+ind.N {
        return errors.New("Sytrf: sizeA")
    }
    if ipiv != nil && len(ipiv) < ind.N {
        return errors.New("Sytrf: size ipiv")
    }
    return nil
}

// Local Variables:
// tab-width: 4
// End:
