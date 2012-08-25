
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package lapack

import (
	"github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/matrix"
	"errors"
	"fmt"
)

/*
 QR factorization.

 Geqrf(A, tau, m=A.Rows, n=A.Cols, ldA=max(1,A.Rows), offsetA=0)

 PURPOSE

 QR factorization of an m by n real or complex matrix A:

  A = Q*R = [Q1 Q2] * [R1; 0] if m >= n
  A = Q*R = Q * [R1 R2]       if m <= n,

 where Q is m by m and orthogonal/unitary and R is m by n with R1
 upper triangular.  On exit, R is stored in the upper triangular
 part of A.  Q is stored as a product of k=min(m,n) elementary
 reflectors.  The parameters of the reflectors are stored in the
 first k entries of tau and in the lower triangular part of the
 first k columns of A.

 ARGUMENTS
  A         float or complex matrix
  tau       float or complex  matrix of length at least min(m,n).  Must
            have the same type as A.
  m         integer.  If negative, the default value is used.
  n         integer.  If negative, the default value is used.
  ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the
            default value is used.
  offsetA   nonnegative integer

 */
func Geqrf(A, tau matrix.Matrix, opts ...linalg.Option) error {
	ind := linalg.GetIndexOpts(opts...)
	if ind.N < 0 {
		ind.N = A.Rows()
	}
	if ind.M < 0 {
		ind.M = A.Cols()
	}
	if ind.N == 0 || ind.M == 0 {
		return nil
	}
	if ind.LDa == 0 {
		ind.LDa = max(1, A.Rows())
	}
	if ind.LDa < max(1, ind.M) {
		return errors.New("Geqrf: ldA")
	}
	if ind.OffsetA < 0 {
		return errors.New("Geqrf: offsetA")
	}
	if A.NumElements() < ind.OffsetA + ind.K*ind.LDa {
		return errors.New("Geqrf: sizeA")
	}
	if tau.NumElements() < min(ind.M, ind.N) {
		return errors.New("Geqrf: sizeTau")
	}
	if ! matrix.EqualTypes(A, tau) {
		return errors.New("Geqrf: arguments not of same type")
	}
	info := -1
	switch A.(type) {
	case *matrix.FloatMatrix:
		Aa := A.FloatArray()
		taua := tau.FloatArray()
		info = dgeqrf(ind.M, ind.N, Aa[ind.OffsetA:], ind.LDa, taua)
	case *matrix.ComplexMatrix:
		return errors.New("Geqrf: complex not yet implemented")
	}
	if info != 0 {
		return errors.New(fmt.Sprintf("Geqrf lapack error: %d", info))
	}
	return nil
}


// Local Variables:
// tab-width: 4
// End:
