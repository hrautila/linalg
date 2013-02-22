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
 Computes selected eigenvalues and eigenvectors of a real symmetric
 matrix (expert driver).

 PURPOSE

 Computes selected eigenvalues/vectors of a real symmetric n by n
 matrix A.

 If range is OptRangeAll, all eigenvalues are computed.
 If range is OptRangeValue, all eigenvalues in the interval (vlimit[0],vlimit[1]] are
 computed.
 If range is OptRangeInt, all eigenvalues il through iu are computed
 (sorted in ascending order with 1 <= il <= iu <= n).

 If jobz is OptJobNo, only the eigenvalues are returned in W.
 If jobz is OptJobValue, the eigenvectors are also returned in Z.

 On exit, the content of A is destroyed.

 ARGUMENTS
  A         float matrix
  W         float matrix of length at least n.  On exit, contains
            the computed eigenvalues in ascending order.
  Z         float matrix.  Only required when jobz is PJobValue.  If range
            is PRangeAll or PRangeValue, Z must have at least n columns.  If
            range is PRangeInt, Z must have at least iu-il+1 columns.
            On exit the first m columns of Z contain the computed (normalized) eigenvectors.
  vlimit    []float64 or nul.  Only required when range is PRangeValue
  ilimit    []int or nil.  Only required when range is PRangeInt.
  abstol    double.  Absolute error tolerance for eigenvalues.
            If nonpositive, the LAPACK default value is used.

 OPTIONS
  jobz      linalg.OptJobNo or linalg.OptJobValue
  range     linalg.OptRangeAll, linalg.OptRangeValue or linalg.OptRangeInt
  uplo      linalg.OptLower or linalg.OptUpper
  n         integer.  If negative, the default value is used.
  m         the number of eigenvalues computed;
  ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the
            default value is used.
  ldZ       nonnegative integer.  ldZ >= 1 if jobz is PJobNo and
            ldZ >= max(1,n) if jobz is PJobValue.  The default value
            is 1 if jobz is PJobNo and max(1,Z.size[0]) if jobz =PJobValue.
            If zero, the default value is used.
  offsetA   nonnegative integer
  offsetW   nonnegative integer
  offsetZ   nonnegative integer
*/
func Syevx(A, W, Z matrix.Matrix, abstol float64, vlimit []float64, ilimit []int, opts ...linalg.Option) error {
    if !matrix.EqualTypes(A, W, Z) {
        return onError("Syevx: not same type")
    }
    switch A.(type) {
    case *matrix.FloatMatrix:
        Am := A.(*matrix.FloatMatrix)
        Wm := W.(*matrix.FloatMatrix)
        var Zm *matrix.FloatMatrix = nil
        if Z != nil {
            Zm = Z.(*matrix.FloatMatrix)
        }
        return SyevrFloat(Am, Wm, Zm, abstol, vlimit, ilimit, opts...)
    }
    return onError("Syevr: unknown types")
}

func SyevxFloat(A, W, Z matrix.Matrix, abstol float64, vlimit []float64, ilimit []int, opts ...linalg.Option) error {
    var vl, vu float64
    var il, iu int

    pars, err := linalg.GetParameters(opts...)
    if err != nil {
        return err
    }
    ind := linalg.GetIndexOpts(opts...)
    arows := ind.LDa
    if ind.N < 0 {
        ind.N = A.Rows()
        if ind.N != A.Cols() {
            return onError("Syevr: A not square")
        }
    }
    // Check indexes
    if ind.N == 0 {
        return nil
    }
    if ind.LDa == 0 {
        ind.LDa = max(1, A.LeadingIndex())
        arows = max(1, A.Rows())
    }
    if ind.LDa < max(1, A.Rows()) {
        return onError("Syevr: lda")
    }
    if pars.Range == linalg.PRangeValue {
        if vlimit == nil {
            return onError("Syevx: vlimit is nil")
        }
        vl = vlimit[0]
        vu = vlimit[1]
        if vl >= vu {
            return onError("Syevx: must be: vl < vu")
        }
    } else if pars.Range == linalg.PRangeInt {
        if ilimit == nil {
            return onError("Syevx: ilimit is nil")
        }
        il = ilimit[0]
        iu = ilimit[1]
        if il < 1 || il > iu || iu > ind.N {
            return onError("Syevx: must be:1 <= il <= iu <= N")
        }
    }
    if pars.Jobz == linalg.PJobValue {
        if Z == nil {
            return onError("Syevx: Z is nil")
        }
        if ind.LDz == 0 {
            ind.LDz = max(1, Z.LeadingIndex())
        }
        if ind.LDz < max(1, ind.N) {
            return onError("Syevx: ldz")
        }
    } else {
        if ind.LDz == 0 {
            ind.LDz = 1
        }
        if ind.LDz < 1 {
            return onError("Syevx: ldz")
        }
    }
    if ind.OffsetA < 0 {
        return onError("Syevx: OffsetA")
    }
    sizeA := A.NumElements()
    if sizeA < ind.OffsetA+(ind.N-1)*arows+ind.N {
        return onError("Syevx: sizeA")
    }
    if ind.OffsetW < 0 {
        return onError("Syevx: OffsetW")
    }
    sizeW := W.NumElements()
    if sizeW < ind.OffsetW+ind.N {
        return onError("Syevx: sizeW")
    }
    if pars.Jobz == linalg.PJobValue {
        if ind.OffsetZ < 0 {
            return onError("Syevx: OffsetW")
        }
        zrows := max(1, Z.Rows())
        minZ := ind.OffsetZ + (ind.N-1)*zrows + ind.N
        if pars.Range == linalg.PRangeInt {
            minZ = ind.OffsetZ + (iu-il)*zrows + ind.N
        }
        if Z.NumElements() < minZ {
            return onError("Syevx: sizeZ")
        }
    }

    Aa := A.(*matrix.FloatMatrix).FloatArray()
    Wa := W.(*matrix.FloatMatrix).FloatArray()
    var Za []float64
    if pars.Jobz == linalg.PJobValue {
        Za = Z.(*matrix.FloatMatrix).FloatArray()
    } else {
        Za = nil
    }
    jobz := linalg.ParamString(pars.Jobz)
    rnge := linalg.ParamString(pars.Range)
    uplo := linalg.ParamString(pars.Uplo)

    info := dsyevx(jobz, rnge, uplo, ind.N, Aa[ind.OffsetA:], ind.LDa,
        vl, vu, il, iu, ind.M, Wa[ind.OffsetW:], Za, ind.LDz)
    if info != 0 {
        return onError(fmt.Sprintf("Syevx: call failed %d", info))
    }
    return nil
}

// Local Variables:
// tab-width: 4
// End:
