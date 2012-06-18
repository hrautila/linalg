
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
 Cholesky factorization of a real symmetric or complex Hermitian
 positive definite matrix.

 Potrf(A, uplo=PLower, n=A.Rows, ldA=max(1,A.Rows), offsetA=0)
 
 PURPOSE

 Factors A as A=L*L^T or A = L*L^H, where A is n by n, real
 symmetric or complex Hermitian, and positive definite.

 On exit, if uplo=PLower, the lower triangular part of A is replaced
 by L.  If uplo=PUpper, the upper triangular part is replaced by L^T
 or L^H.

 ARGUMENTS
  A         float or complex matrix

 OPTIONS
  uplo      PLower or PUpper
  n         nonnegative integer.  If negative, the default value is used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default
            value is used.
  offsetA   nonnegative integer

 */
func Potrf(A matrix.Matrix, opts ...linalg.Option) error {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return err
	}
	ind := linalg.GetIndexOpts(opts...)
	if ind.N < 0 {
		ind.N = A.Rows()
		if ind.N != A.Cols() {
			return errors.New("not square")
		}
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
		info = dpotrf(uplo, ind.N, Aa[ind.OffsetA:], ind.LDa)
	case *matrix.ComplexMatrix:
		return errors.New("ComplexMatrx: not implemented yet")
	}
	if info != 0 {
		return errors.New("Potrf failed")
	}
	return nil
}


// Local Variables:
// tab-width: 4
// End:
