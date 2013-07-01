// Copyright (c) Harri Rautila, 2012, 2013

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
 Solves a real or complex set of linear equations with a banded
 coefficient matrix.

 PURPOSE

 Solves A*X = B

 A an n by n real or complex band matrix with kl subdiagonals and
 ku superdiagonals.

 If ipiv is provided, then on entry the kl+ku+1 diagonals of the
 matrix are stored in rows kl+1 to 2*kl+ku+1 of A, in the BLAS
 format for general band matrices.  On exit, A and ipiv contain the
 details of the factorization.  If ipiv is not provided, then on
 entry the diagonals of the matrix are stored in rows 1 to kl+ku+1 
 of A, and Gbsv() does not return the factorization and does not
 modify A.  On exit B is replaced with solution X.

 ARGUMENTS.
  A         float or complex banded matrix
  B         float or complex matrix.  Must have the same type as A.
  kl        nonnegative integer
  ipiv      int array of length at least n

 OPTIONS
  ku        nonnegative integer.  If negative, the default value is
            used.  The default value is A.Rows-kl-1 if ipiv is
            not provided, and A.Rows-2*kl-1 otherwise.
  n         nonnegative integer.  If negative, the default value is used.
  nrhs      nonnegative integer.  If negative, the default value is used.
  ldA       positive integer.  ldA >= kl+ku+1 if ipiv is not provided
            and ldA >= 2*kl+ku+1 if ipiv is provided.  If zero, the
            default value is used.
  ldB       positive integer.  ldB >= max(1,n).  If zero, the default
            default value is used.
  offsetA   nonnegative integer
  offsetB   nonnegative integer;

*/
func Gbsv(A, B matrix.Matrix, ipiv []int32, kl int, opts ...linalg.Option) error {
	if !matrix.EqualTypes(A, B) {
		return onError("Gbsv: not same type")
	}
	switch A.(type) {
	case *matrix.FloatMatrix:
		Am := A.(*matrix.FloatMatrix)
		Bm := B.(*matrix.FloatMatrix)
		return GbsvFloat(Am, Bm, ipiv, kl, opts...)
	case *matrix.ComplexMatrix:
		Am := A.(*matrix.ComplexMatrix)
		Bm := B.(*matrix.ComplexMatrix)
		return GbsvComplex(Am, Bm, ipiv, kl, opts...)
	}
	return onError("Gbsv: unknown types types!")
}

func GbsvFloat(A, B *matrix.FloatMatrix, ipiv []int32, kl int, opts ...linalg.Option) error {

	ind := linalg.GetIndexOpts(opts...)
	ind.Kl = kl
	err := checkGbsv(ind, A, B, ipiv)
	if err != nil {
		return err
	}
	if ind.N == 0 || ind.Nrhs == 0 {
		return nil
	}

	Aa := A.FloatArray()
	Ba := B.FloatArray()
	info := dgbsv(ind.N, ind.Kl, ind.Ku, ind.Nrhs, Aa[ind.OffsetA:], ind.LDa,
		ipiv, Ba[ind.OffsetB:], ind.LDb)
	if info != 0 {
		return onError(fmt.Sprintf("Gbsv lapack error: %d", info))
	}
	return nil
}

func GbsvComplex(A, B *matrix.ComplexMatrix, ipiv []int32, kl int, opts ...linalg.Option) error {
	ind := linalg.GetIndexOpts(opts...)
	ind.Kl = kl
	err := checkGbsv(ind, A, B, ipiv)
	if err != nil {
		return err
	}
	if ind.N == 0 || ind.Nrhs == 0 {
		return nil
	}
	return onError("Gbsv: complex not implemented yet")
}

func checkGbsv(ind *linalg.IndexOpts, A, B matrix.Matrix, ipiv []int32) error {
	arows := ind.LDa
	brows := ind.LDb
	if ind.Kl < 0 {
		return onError("Gbsv: invalid kl")
	}
	if ind.N < 0 {
		ind.N = A.Rows()
	}
	if ind.Nrhs < 0 {
		ind.Nrhs = A.Cols()
	}
	if ind.N == 0 || ind.Nrhs == 0 {
		return nil
	}
	if ind.Ku < 0 {
		ind.Ku = A.Rows() - 2*ind.Kl - 1
	}
	if ind.Ku < 0 {
		return onError("Gbsv: invalid ku")
	}
	if ind.LDa == 0 {
		ind.LDa = max(1, A.LeadingIndex())
		arows = max(1, A.Rows())
	}
	if ind.LDa < 2*ind.Kl+ind.Ku+1 {
		return onError("Gbsv: lda")
	}
	if ind.OffsetA < 0 {
		return onError("Gbsv: offsetA")
	}
	sizeA := A.NumElements()
	if sizeA < ind.OffsetA+(ind.N-1)*arows+2*ind.Kl+ind.Ku+1 {
		return onError("Gbsv: sizeA")
	}
	if ind.LDb == 0 {
		ind.LDb = max(1, B.LeadingIndex())
		brows = max(1, B.Rows())
	}
	if ind.OffsetB < 0 {
		return onError("Gbsv: offsetB")
	}
	sizeB := B.NumElements()
	if sizeB < ind.OffsetB+(ind.Nrhs-1)*brows+ind.N {
		return onError("Gbsv: sizeB")
	}
	if ipiv != nil && len(ipiv) < ind.N {
		return onError("Gbsv: size ipiv")
	}
	return nil
}

// Local Variables:
// tab-width: 4
// End:
