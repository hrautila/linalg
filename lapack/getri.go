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
 Inverse of a real or complex matrix.

 PURPOSE

 Computes the inverse of real or complex matrix of order n.  On
 entry, A and ipiv contain the LU factorization, as returned by
 gesv() or getrf().  On exit A is replaced by the inverse.

 ARGUMENTS
  A         float or complex matrix
  ipiv      int vector

 OPTIONS
  n         nonnegative integer.  If negative, the default value is used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default
            value is used.
  offsetA   nonnegative integer;
*/
func Getri(A matrix.Matrix, ipiv []int32, opts ...linalg.Option) error {
    ind := linalg.GetIndexOpts(opts...)
    arows := ind.LDa
    if ind.N < 0 {
        ind.N = A.Cols()
    }
    if ind.N == 0 {
        return nil
    }
    if ind.LDa == 0 {
        ind.LDa = max(1, A.LeadingIndex())
        arows = max(1, A.Rows())
    }
    if ind.OffsetA < 0 {
        return onError("Getri: offset")
    }
    sizeA := A.NumElements()
    if sizeA < ind.OffsetA+(ind.N-1)*arows+ind.N {
        return onError("Getri: sizeA")
    }
    if ipiv != nil && len(ipiv) < ind.N {
        return onError("Getri: size ipiv")
    }
    info := -1
    switch A.(type) {
    case *matrix.FloatMatrix:
        Aa := A.(*matrix.FloatMatrix).FloatArray()
        info = dgetri(ind.N, Aa[ind.OffsetA:], ind.LDa, ipiv)
    case *matrix.ComplexMatrix:
        return onError("Getri: complex not yet implemented")
    }
    if info != 0 {
        return onError(fmt.Sprintf("Getri lapack error: %d", info))
    }
    return nil
}

// Local Variables:
// tab-width: 4
// End:
