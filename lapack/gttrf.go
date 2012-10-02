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
 LU factorization of a real or complex tridiagonal matrix.

 PURPOSE

 Factors an n by n real or complex tridiagonal matrix A as A = P*L*U.

 A is specified by its lower diagonal dl, diagonal d, and upper
 diagonal du.  On exit dl, d, du, du2 and ipiv contain the details
 of the factorization.

 ARGUMENTS.
  DL        float or complex matrix
  D         float or complex matrix.  Must have the same type as DL.
  DU        float or complex matrix.  Must have the same type as DL.
  DU2       float or complex matrix of length at least n-2.  Must have the
            same type as DL.
  ipiv      int vector of length at least n

 OPTIONS
  n         nonnegative integer.  If negative, the default value is used.
  offsetdl  nonnegative integer
  offsetd   nonnegative integer
  offsetdu  nonnegative integer
*/
func Gtrrf(DL, D, DU, DU2 matrix.Matrix, ipiv []int32, opts ...linalg.Option) error {
    ind := linalg.GetIndexOpts(opts...)
    if ind.OffsetD < 0 {
        return errors.New("Gttrf: offset D")
    }
    if ind.N < 0 {
        ind.N = D.NumElements() - ind.OffsetD
    }
    if ind.N < 0 {
        return errors.New("Gttrf: size D")
    }
    if ind.N == 0 {
        return nil
    }
    if ind.OffsetDL < 0 {
        return errors.New("Gttrf: offset DL")
    }
    sizeDL := DL.NumElements()
    if sizeDL < ind.OffsetDL+ind.N-1 {
        return errors.New("Gttrf: sizeDL")
    }
    if ind.OffsetDU < 0 {
        return errors.New("Gttrf: offset DU")
    }
    sizeDU := DU.NumElements()
    if sizeDU < ind.OffsetDU+ind.N-1 {
        return errors.New("Gttrf: sizeDU")
    }
    sizeDU2 := DU2.NumElements()
    if sizeDU2 < ind.N-2 {
        return errors.New("Gttrf: sizeDU2")
    }
    if len(ipiv) < ind.N {
        return errors.New("Gttrf: size ipiv")
    }
    info := -1
    if !matrix.EqualTypes(DL, D, DU, DU2) {
        return errors.New("Gttrf: arguments not same type")
    }
    switch DL.(type) {
    case *matrix.FloatMatrix:
        DLa := DL.(*matrix.FloatMatrix).FloatArray()
        Da := D.(*matrix.FloatMatrix).FloatArray()
        DUa := DU.(*matrix.FloatMatrix).FloatArray()
        DU2a := DU2.(*matrix.FloatMatrix).FloatArray()
        info = dgttrf(ind.N, DLa[ind.OffsetDL:], Da[ind.OffsetD:], DUa[ind.OffsetDU:],
            DU2a, ipiv)
    case *matrix.ComplexMatrix:
        return errors.New("Gttrf: complex not yet implemented")
    }
    if info != 0 {
        return errors.New(fmt.Sprintf("Gttrf lapack error: %d", info))
    }
    return nil
}

// Local Variables:
// tab-width: 4
// End:
