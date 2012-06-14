
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
 Solves a real symmetric or complex Hermitian positive definite set
 of linear equations, given the Cholesky factorization computed by
 potrf() or posv().

 Potrs(A, B, uplo=PLower, n=A.Rows, nrhs=B.Cols,
 ldA=max(1,A.Rows), ldB=max(1,B.Rows), offsetA=0, offsetB=0)

 PURPOSE

 Solves
   A*X = B

 where A is n by n, real symmetric or complex Hermitian and positive definite,
 and B is n by nrhs. On entry, A contains the Cholesky factor, as
 returned by Posv() or Potrf().  On exit B is replaced by the solution X.

 ARGUMENTS
  A         float or complex matrix
  B         float or complex matrix.  Must have the same type as A.

 OPTIONS
  uplo      PLower or PUpper
  n         nonnegative integer.  If negative, the default value is used.
  nrhs      nonnegative integer.  If negative, the default value is used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default
            value is used.
  ldB       positive integer.  ldB >= max(1,n).  If zero, the default
            value is used.
  offsetA   nonnegative integer
  offsetB   nonnegative integer;

 */
func Potrs(A, B matrix.Matrix, opts ...linalg.Opt) error {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return err
	}
	ind := linalg.GetIndexOpts(opts...)
	if ind.N < 0 {
		ind.N = A.Rows()
	}
	if ind.Nrhs < 0 {
		ind.Nrhs = B.Cols()
	}
	if ind.N == 0 || ind.Nrhs == 0 {
		return nil
	}
	if ind.LDa == 0 {
		ind.LDa = max(1, A.Rows())
	}
	if ind.LDa < max(1, ind.N) {
		return errors.New("lda")
	}
	if ind.LDb == 0 {
		ind.LDb = max(1, B.Rows())
	}
	if ind.LDb < max(1, ind.N) {
		return errors.New("ldb")
	}
	if ind.OffsetA < 0 {
		return errors.New("offsetA")
	}
	if A.NumElements() < ind.OffsetA + (ind.N-1)*ind.LDa + ind.N {
		return errors.New("sizeA")
	}
	if ind.OffsetB < 0 {
		return errors.New("offsetB")
	}
	if B.NumElements() < ind.OffsetB + (ind.Nrhs-1)*ind.LDb + ind.N {
		return errors.New("sizeB")
	}
	if ! matrix.EqualTypes(A, B) {
		return errors.New("types")
	}
	info := -1
	switch A.(type) {
	case *matrix.FloatMatrix:
		Aa := A.FloatArray()
		Ba := B.FloatArray()
		uplo := linalg.ParamString(pars.Uplo)
		info = dpotrs(uplo, ind.N, ind.Nrhs, Aa[ind.OffsetA:], ind.LDa,
			Ba[ind.OffsetB:], ind.LDb)
	case *matrix.ComplexMatrix:
		return errors.New("ComplexMatrx: not implemented yet")
	}
	if info != 0 {
		return errors.New("Potrs failed")
	}
	return nil
}

// Local Variables:
// tab-width: 4
// End:
