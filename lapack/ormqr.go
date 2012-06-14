
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package lapack

import (
	"linalg"
	"matrix"
	"errors"
)

/*
 Product with a real orthogonal matrix.

 Ormqr(A, tau, C, side='L', trans='N', m=C.Rows, n=C.Cols,
 k=len(tau), ldA=max(1,A.Rows), ldC=max(1,C.Rows), offsetA=0, offsetC=0)

 PURPOSE

 Computes
  C := Q*C   if side = PLeft  and trans = PNoTrans
  C := Q^T*C if side = PLeft  and trans = PTrans
  C := C*Q   if side = PRight and trans = PNoTrans
  C := C*Q^T if side = PRight and trans = PTrans

 C is m by n and Q is a square orthogonal matrix computed by geqrf.
 
 Q is defined as a product of k elementary reflectors, stored as
 the first k columns of A and the first k entries of tau.

 ARGUMENTS
  A         float matrix
  tau       float matrix of length at least k
  C         float matrix
 
 OPTIONS
  side      PLeft or PRight
  trans     PNoTrans or PTrans
  m         integer.  If negative, the default value is used.
  n         integer.  If negative, the default value is used.
  k         integer.  k <= m if side = PRight and k <= n if side = PLeft.
            If negative, the default value is used.
  ldA       nonnegative integer.  ldA >= max(1,m) if side = PLeft
            and ldA >= max(1,n) if side = PRight.  If zero, the
            default value is used.
  ldC       nonnegative integer.  ldC >= max(1,m).  If zero, the
            default value is used.
  offsetA   nonnegative integer
  offsetB   nonnegative integer

*/
func Ormqf(A, tau, C matrix.Matrix, opts ...linalg.Opt) error {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return err
	}
	ind := linalg.GetIndexOpts(opts...)
	if ind.N < 0 {
		ind.N = C.Cols()
	}
	if ind.M < 0 {
		ind.M = C.Rows()
	}
	if ind.K < 0 {
		ind.K = tau.NumElements()
	}
	if ind.N == 0 || ind.M == 0 || ind.K == 0 {
		return nil
	}
	if ind.LDa == 0 {
		ind.LDa = max(1, A.Rows())
	}
	if ind.LDc == 0 {
		ind.LDc = max(1, C.Rows())
	}
	switch pars.Side {
	case linalg.PLeft:
		if ind.K > ind.M {
			errors.New("K")
		}
		if ind.LDa < max(1, ind.M) {
			return errors.New("lda")
		}
	case linalg.PRight:
		if ind.K > ind.N {
			errors.New("K")
		}
		if ind.LDa < max(1, ind.N) {
			return errors.New("lda")
		}
	}
	if ind.OffsetA < 0 {
		return errors.New("offsetA")
	}
	if A.NumElements() < ind.OffsetA + ind.K*ind.LDa {
		return errors.New("sizeA")
	}
	if ind.OffsetC < 0 {
		return errors.New("offsetC")
	}
	if C.NumElements() < ind.OffsetC + (ind.N-1)*ind.LDa + ind.M {
		return errors.New("sizeC")
	}
	if tau.NumElements() < ind.K {
		return errors.New("sizeTau")
	}
	if ! matrix.EqualTypes(A, C, tau) {
		return errors.New("not same type")
	}
	info := -1
	side := linalg.ParamString(pars.Side)
	trans := linalg.ParamString(pars.Trans)
	switch A.(type) {
	case *matrix.FloatMatrix:
		Aa := A.FloatArray()
		Ca := C.FloatArray()
		taua := tau.FloatArray()
		info = dormqr(side, trans, ind.M, ind.N, ind.K, Aa[ind.OffsetA:], ind.LDa,
			taua, Ca[ind.OffsetC:], ind.LDc)
	case *matrix.ComplexMatrix:
		return errors.New("ComplexMatrx: not implemented yet")
	}
	if info != 0 {
		return errors.New("Ormqr failed")
	}
	return nil
}



// Local Variables:
// tab-width: 4
// End:
