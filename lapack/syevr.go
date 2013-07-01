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
 matrix (RRR driver).

 PURPOSE

 Computes selected eigenvalues/vectors of a real symmetric n by n
 matrix A.

 If range is PRangeAll, all eigenvalues are computed.
 If range is PRangeV all eigenvalues in the interval (vlimit[0],vlimit[1]] are
 computed.
 If range is PRangeI, all eigenvalues ilimit[0] through ilimit[1] are computed
 (sorted in ascending order with 1 <= ilimit[0] <= ilimit[1] <= n).

 If jobz is PJobNo, only the eigenvalues are returned in W.
 If jobz is PJobV, the eigenvectors are also returned in Z.
 On exit, the content of A is destroyed.

 Syevr is usually the fastest of the four eigenvalue routines.

 ARGUMENTS
  A         float matrix
  W         float matrix of length at least n.  On exit, contains
            the computed eigenvalues in ascending order.
  Z         float matrix or nil.  Only required when jobz = PJobV.
            If range is PRangeAll or PRangeV, Z must have at least n columns.
            If range is PRangeI, Z must have at least iu-il+1 columns.
            On exit the first m columns of Z contain the computed
            (normalized) eigenvectors.
  abstol    double.  Absolute error tolerance for eigenvalues.
            If nonpositive, the LAPACK default value is used.
  vlmit     []float or nil.  Only required when range is PRangeV.
  ilimit    []int or nil.  Only required when range is PRangeI.

 OPTIONS
  jobz      PJobNo or PJobV
  range     PRangeAll, PRangeV or PRangeI
  uplo      PLower or PUpper
  n         integer.  If negative, the default value is used.
  ldA       nonnegative integer.  ldA >= max(1,n).
            If zero, the default value is used.
  ldZ       nonnegative integer.  ldZ >= 1 if jobz is 'N' and
            ldZ >= max(1,n) if jobz is PJobV.  The default value
            is 1 if jobz is PJobNo and max(1,Z.Rows) if jobz =PJboV.
            If zero, the default value is used.
  offsetA   nonnegative integer
  offsetW   nonnegative integer
  offsetZ   nonnegative integer
  m         the number of eigenvalues computed

*/
func Syevr(A, W, Z matrix.Matrix, abstol float64, vlimit []float64, ilimit []int, opts ...linalg.Option) error {
	if !matrix.EqualTypes(A, W, Z) {
		return onError("Syevr: arguments not of same type")
	}
	switch A.(type) {
	case *matrix.FloatMatrix:
		Am := A.(*matrix.FloatMatrix)
		Wm := W.(*matrix.FloatMatrix)
		Zm := Z.(*matrix.FloatMatrix)
		return SyevrFloat(Am, Wm, Zm, abstol, vlimit, ilimit, opts...)
	}
	return onError("Syevr: unknown types")
}

func SyevrFloat(A, W, Z matrix.Matrix, abstol float64, vlimit []float64, ilimit []int, opts ...linalg.Option) error {
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
			return onError("Syevr: vlimit is nil")
		}
		vl = vlimit[0]
		vu = vlimit[1]
		if vl >= vu {
			return onError("Syevr: must be: vl < vu")
		}
	} else if pars.Range == linalg.PRangeInt {
		if ilimit == nil {
			return onError("Syevr: ilimit is nil")
		}
		il = ilimit[0]
		iu = ilimit[1]
		if il < 1 || il > iu || iu > ind.N {
			return onError("Syevr: must be:1 <= il <= iu <= N")
		}
	}
	if pars.Jobz == linalg.PJobValue {
		if Z == nil {
			return onError("Syevr: Z is nil")
		}
		if ind.LDz == 0 {
			ind.LDz = max(1, Z.LeadingIndex())
		}
		if ind.LDz < max(1, ind.N) {
			return onError("Syevr: ldz")
		}
	} else {
		if ind.LDz == 0 {
			ind.LDz = 1
		}
		if ind.LDz < 1 {
			return onError("Syevr: ldz")
		}
	}
	if ind.OffsetA < 0 {
		return onError("Syevr: OffsetA")
	}
	sizeA := A.NumElements()
	if sizeA < ind.OffsetA+(ind.N-1)*arows+ind.N {
		return onError("Syevr: sizeA")
	}
	if ind.OffsetW < 0 {
		return onError("Syevr: OffsetW")
	}
	sizeW := W.NumElements()
	if sizeW < ind.OffsetW+ind.N {
		return onError("Syevr: sizeW")
	}
	if pars.Jobz == linalg.PJobValue {
		if ind.OffsetZ < 0 {
			return onError("Syevr: OffsetW")
		}
		zrows := max(1, Z.Rows())
		minZ := ind.OffsetZ + (ind.N-1)*zrows + ind.N
		if pars.Range == linalg.PRangeInt {
			minZ = ind.OffsetZ + (iu-il)*zrows + ind.N
		}
		if Z.NumElements() < minZ {
			return onError("Syevr: sizeZ")
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

	info := dsyevr(jobz, rnge, uplo, ind.N, Aa[ind.OffsetA:], ind.LDa,
		vl, vu, il, iu, ind.M, Wa[ind.OffsetW:], Za, ind.LDz)
	if info != 0 {
		return onError(fmt.Sprintf("Syevr: lapack error %d", info))
	}
	return nil
}

// Local Variables:
// tab-width: 4
// End:
