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
 Solves a real or complex tridiagonal set of linear equations, 
 given the LU factorization computed by gttrf().

 PURPOSE
  solves A*X=B,   if trans is PNoTrans
  solves A^T*X=B, if trans is PTrans
  solves A^H*X=B, if trans is PConjTrans

 On entry, DL, D, DU, DU2 and ipiv contain the LU factorization of 
 an n by n tridiagonal matrix A as computed by gttrf().  On exit B
 is replaced by the solution X.

 ARGUMENTS.
  DL        float or complex matrix
  D         float or complex matrix.  Must have the same type as dl.
  DU        float or complex matrix.  Must have the same type as dl.
  DU2       float or complex matrix.  Must have the same type as dl.
  B         float or complex matrix.  Must have the same type oas dl.
  ipiv      int vector

 OPTIONS
  trans     PNoTrans, PTrans, PConjTrans
  n         nonnegative integer.  If negative, the default value is used.
  nrhs      nonnegative integer.  If negative, the default value is used.
  ldB       positive integer, ldB >= max(1,n). If zero, the default value is used.
  offsetdl  nonnegative integer
  offsetd   nonnegative integer
  offsetdu  nonnegative integer
  offsetB   nonnegative integer

*/
func Gtrrs(DL, D, DU, DU2, B matrix.Matrix, ipiv []int32, opts ...linalg.Option) error {
    pars, err := linalg.GetParameters(opts...)
    if err != nil {
        return err
    }
    ind := linalg.GetIndexOpts(opts...)
    if ind.OffsetD < 0 {
        return errors.New("Gttrs: offset D")
    }
    if ind.N < 0 {
        ind.N = D.NumElements() - ind.OffsetD
    }
    if ind.N < 0 {
        return errors.New("Gttrs: size D")
    }
    if ind.N == 0 {
        return nil
    }
    if ind.OffsetDL < 0 {
        return errors.New("Gttrs: offset DL")
    }
    sizeDL := DL.NumElements()
    if sizeDL < ind.OffsetDL+ind.N-1 {
        return errors.New("Gttrs: sizeDL")
    }
    if ind.OffsetDU < 0 {
        return errors.New("Gttrs: offset DU")
    }
    sizeDU := DU.NumElements()
    if sizeDU < ind.OffsetDU+ind.N-1 {
        return errors.New("Gttrs: sizeDU")
    }
    sizeDU2 := DU2.NumElements()
    if sizeDU2 < ind.N-2 {
        return errors.New("Gttrs: sizeDU2")
    }
    if ind.Nrhs < 0 {
        ind.Nrhs = B.Cols()
    }
    if ind.Nrhs == 0 {
        return nil
    }
    if ind.LDb == 0 {
        ind.LDb = max(1, B.LeadingIndex())
    }
    if ind.LDb < max(1, ind.N) {
        return errors.New("Gttrs: ldB")
    }
    if ind.OffsetB < 0 {
        return errors.New("Gttrs: offset B")
    }
    sizeB := B.NumElements()
	brows := max(1, B.Rows())
    if sizeB < ind.OffsetB+(ind.Nrhs-1)*brows+ind.N {
        return errors.New("Gttrs: sizeB")
    }
    if len(ipiv) < ind.N {
        return errors.New("Gttrs: size ipiv")
    }
    if !matrix.EqualTypes(DL, D, DU, DU2, B) {
        return errors.New("Gttrs: matrix types")
    }
    var info int = -1
    switch DL.(type) {
    case *matrix.FloatMatrix:
        DLa := DL.(*matrix.FloatMatrix).FloatArray()
        Da := D.(*matrix.FloatMatrix).FloatArray()
        DUa := DU.(*matrix.FloatMatrix).FloatArray()
        DU2a := DU2.(*matrix.FloatMatrix).FloatArray()
        Ba := B.(*matrix.FloatMatrix).FloatArray()
        trans := linalg.ParamString(pars.Trans)
        info = dgttrs(trans, ind.N, ind.Nrhs,
            DLa[ind.OffsetDL:], Da[ind.OffsetD:], DUa[ind.OffsetDU:], DU2a,
            ipiv, Ba[ind.OffsetB:], ind.LDb)
    case *matrix.ComplexMatrix:
        return errors.New("Gttrs: complex valued not yet implemented")
    }
    if info != 0 {
        return errors.New(fmt.Sprintf("Gttrs lapack error: %d", info))
    }
    return nil
}

// Local Variables:
// tab-width: 4
// End:
