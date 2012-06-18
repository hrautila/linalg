
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
 LU factorization of a real or complex m by n band matrix.

 Gbtrf(A, m, kl, ipiv, n=A.Cols, ku=A.Rows-2*kl-1, ldA=max(1,A.Rows), offsetA=0)

 PURPOSE

 Computes the LU factorization of an m by n band matrix with kl
 subdiagonals and ku superdiagonals.  On entry, the diagonals are
 stored in rows kl+1 to 2*kl+ku+1 of the array A, in the BLAS format
 for general band matrices.   On exit A and ipiv contains the
 factorization.

 ARGUMENTS
  A         float or complex matrix
  ipiv      int array of length at least min(m,n)
  m         nonnegative integer
  kl        nonnegative integer.

 OPTIONS
  n         nonnegative integer, default =A.Cols()
  ku        nonnegative integer. default = A.Rows()-2*kl+1
  ldA       positive integer, ldA >= 2*kl+ku+1. default = min(1, A.Rows())
  offsetA   nonnegative integer
 */
func Gbtrf(A matrix.Matrix, ipiv []int32, M, KL int, opts ...linalg.Option) error {
	ind := linalg.GetIndexOpts(opts...)
	ind.M = M
	ind.Kl = KL
	if ind.M < 0 {
		return errors.New("illegal m")
	}
	if ind.Kl < 0 {
		return errors.New("illegal kl")
	}
	if ind.N < 0 {
		ind.N = A.Rows()
	}
	if ind.M == 0 || ind.N == 0 {
		return nil
	}
	if ind.Ku < 0 {
		ind.Ku = A.Rows() - 2*ind.Kl - 1
	}
	if ind.Ku < 0 {
		return errors.New("invalid ku")
	}
	if ind.LDa == 0 {
		ind.LDa = max(1, A.Rows())
	}
	if ind.LDa < 2*ind.Kl + ind.Ku + 1 {
		return errors.New("lda")
	}
	if ind.OffsetA < 0 {
		return errors.New("offsetA")
	}
	sizeA := A.NumElements()
	if sizeA < ind.OffsetA+(ind.N-1)*ind.LDa + 2*ind.Kl + ind.Ku + 1 {
		return errors.New("sizeA")
	}
	if ipiv != nil && len(ipiv) < min(ind.N, ind.M) {
		return errors.New("size ipiv")
	}
	info := -1
	switch A.(type) {
	case *matrix.FloatMatrix:
		Aa := A.FloatArray()
		info = dgbtrf(ind.M, ind.N, ind.Kl, ind.Ku, Aa[ind.OffsetA:], ind.LDa, ipiv)
	case *matrix.ComplexMatrix:
		return errors.New("complex not yet implemented.")
	}
	if info != 0 {
		return errors.New("gbtrf call error")
	}
	return nil
}


// Local Variables:
// tab-width: 4
// End:
