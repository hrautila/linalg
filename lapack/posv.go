
// Copyright (c) Harri Rautila, 2012

// This file is part of github.com/hrautila/linalg/lapack package.
// It is free software, distributed under the terms of GNU Lesser General Public 
// License Version 3, or any later version. See the COPYING tile included in this archive.

package lapack

import (
	"github.com/hrautila/linalg"
	"github.com/hrautila/matrix"
	"errors"
	"fmt"
)
/*
 Solves a real symmetric or complex Hermitian positive definite set
 of linear equations.

 PURPOSE

 Solves A*X = B with A n by n, real symmetric or complex Hermitian,
 and positive definite, and B n by nrhs.
 On exit, if uplo is PLower,  the lower triangular part of A is
 replaced by L.  If uplo is PUpper, the upper triangular part is
 replaced by L^H.  B is replaced by the solution.

 ARGUMENTS.
  A         float or complex matrix
  B         float or complex matrix.  Must have the same type as A.

 OPTIONS
  uplo      PLower or PUpper
  n         nonnegative integer.  If negative, the default value is used.
  nrhs      nonnegative integer.  If negative, the default value is  used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default value is used.
  ldB       positive integer.  ldB >= max(1,n).  If zero, the default value is used.
  offsetA   nonnegative integer
  offsetB   nonnegative integer
 */
func Posv(A, B matrix.Matrix, opts ...linalg.Option) error {
	if ! matrix.EqualTypes(A, B) {
		return errors.New("Posv: arguments not same type")
	}
	switch A.(type) {
	case *matrix.FloatMatrix:
		Am := A.(*matrix.FloatMatrix)
		Bm := B.(*matrix.FloatMatrix)
		return PosvFloat(Am, Bm, opts...)
	case *matrix.ComplexMatrix:
		Am := A.(*matrix.ComplexMatrix)
		Bm := B.(*matrix.ComplexMatrix)
		return PosvComplex(Am, Bm, opts...)
	}
	return errors.New("Posv: unknown types")
}

func PosvFloat(A, B *matrix.FloatMatrix, opts ...linalg.Option) error {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return err
	}
	ind := linalg.GetIndexOpts(opts...)
	err = checkPosv(ind, A, B)
	if err != nil {
		return err
	}
	if ind.N == 0 || ind.Nrhs == 0 {
		return nil
	}
	Aa := A.FloatArray()
	Ba := B.FloatArray()
	uplo := linalg.ParamString(pars.Uplo)
	info := dposv(uplo, ind.N, ind.Nrhs, Aa[ind.OffsetA:], ind.LDa, Ba[ind.OffsetB:], ind.LDb)
	if info != 0 {
		return errors.New(fmt.Sprintf("Posv: lapack error %d", info))
	}
	return nil
}

func PosvComplex(A, B *matrix.ComplexMatrix, opts ...linalg.Option) error {
	return errors.New(fmt.Sprintf("Posv: complex not yet implemented"))
}

func checkPosv(ind *linalg.IndexOpts, A, B matrix.Matrix) error {
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
		return errors.New("Posv: lda")
	}
	if ind.LDb == 0 {
		ind.LDb = max(1, B.Rows())
	}
	if ind.LDb < max(1, ind.N) {
		return errors.New("Posv: ldb")
	}
	if ind.OffsetA < 0 {
		return errors.New("Posv: offsetA")
	}
	sizeA := A.NumElements()
	if sizeA < ind.OffsetA + (ind.N-1)*ind.LDa + ind.N {
		return errors.New("Posv: sizeA")
	}
	if ind.OffsetB < 0 {
		return errors.New("Posv: offsetB")
	}
	sizeB := B.NumElements()
	if sizeB < ind.OffsetB + (ind.Nrhs-1)*ind.LDb + ind.N {
		return errors.New("Posv: sizeB")
	}
	return nil
}
// Local Variables:
// tab-width: 4
// End:
