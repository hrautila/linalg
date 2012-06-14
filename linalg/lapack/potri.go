
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
 Inverse of a real symmetric or complex Hermitian positive definite
 matrix.

 Potri(A, uplo=PLower, n=A.Rows, ldA=max(1,A.Rows), offsetA=0)

 PURPOSE

 Computes the inverse of a real symmetric or complex Hermitian
 positive definite matrix of order n.  On entry, A contains the
 Cholesky factor, as returned by posv() or potrf().  On exit it is
 replaced by the inverse.

 ARGUMENTS
  A         float or complex matrix

 OPTIONS
  uplo      PLower orPUpper
  n         nonnegative integer.  If negative, the default value is  used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default
            value is used.
  offsetA   nonnegative integer;
*/
func Potri(A matrix.Matrix, opts ...linalg.Opt) error {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return err
	}
	ind := linalg.GetIndexOpts(opts...)
	if ind.N < 0 {
		ind.N = A.Rows()
	}
	if ind.N == 0 {
		return nil
	}
	if ind.LDa == 0 {
		ind.LDa = max(1, A.Rows())
	}
	if ind.LDa < max(1, ind.N) {
		return errors.New("lda")
	}
	if ind.OffsetA < 0 {
		return errors.New("offsetA")
	}
	if A.NumElements() < ind.OffsetA + (ind.N-1)*ind.LDa + ind.N {
		return errors.New("sizeA")
	}
	info := -1
	switch A.(type) {
	case *matrix.FloatMatrix:
		Aa := A.FloatArray()
		uplo := linalg.ParamString(pars.Uplo)
		info = dpotri(uplo, ind.N, Aa[ind.OffsetA:], ind.LDa)
	case *matrix.ComplexMatrix:
		return errors.New("ComplexMatrx: not implemented yet")
	}
	if info != 0 {
		return errors.New("Potri failed")
	}
	return nil
}


// Local Variables:
// tab-width: 4
// End:
