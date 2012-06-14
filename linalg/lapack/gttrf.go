
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package lapack

import (
	"github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/matrix"
	"errors"
)

/*
 LU factorization of a real or complex tridiagonal matrix.

 Gttrf(dl, d, du, du2, ipiv, n=len(d)-offsetd, offsetdl=0, offsetd=0, offsetdu=0)

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
func Gtrrf(DL, D, DU, DU2 matrix.Matrix, ipiv []int32, opts ...linalg.Opt) error {
	ind := linalg.GetIndexOpts(opts...)
	if ind.OffsetD < 0 {
		return errors.New("offset D")
	}
	if ind.N < 0 {
		ind.N = D.NumElements() - ind.OffsetD
	}
	if ind.N < 0 {
		return errors.New("size D")
	}
	if ind.N == 0 {
		return nil
	}
	if ind.OffsetDL < 0 {
		return errors.New("offset DL")
	}
	sizeDL := DL.NumElements()
	if sizeDL < ind.OffsetDL + ind.N - 1 {
		return errors.New("sizeDL")
	}
	if ind.OffsetDU < 0 {
		return errors.New("offset DU")
	}
	sizeDU := DU.NumElements()
	if sizeDU < ind.OffsetDU + ind.N - 1 {
		return errors.New("sizeDU")
	}
	sizeDU2 := DU2.NumElements()
	if sizeDU2 < ind.N - 2 {
		return errors.New("sizeDU2")
	}
	if len(ipiv) < ind.N {
		return errors.New("size ipiv")
	}
	info := -1
	switch DL.(type) {
	case *matrix.FloatMatrix:
		DLa := DL.FloatArray()
		Da := D.FloatArray()
		DUa := DU.FloatArray()
		DU2a := DU2.FloatArray()
		info = dgttrf(ind.N, DLa[ind.OffsetDL:], Da[ind.OffsetD:], DUa[ind.OffsetDU:],
			DU2a, ipiv)
	case *matrix.ComplexMatrix:
	}
	if info != 0 {
		return errors.New("Gttrf call error")
	}
	return nil
}

// Local Variables:
// tab-width: 4
// End:
