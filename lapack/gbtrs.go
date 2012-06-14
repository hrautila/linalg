
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
 Solves a real or complex set of linear equations with a banded
 coefficient matrix, given the LU factorization computed by gbtrf()
 or gbsv().

 Gbtrs(A, B, ipiv, kl, trans=PNoTrans, n=A.Cols, ku=A.Rows-2*kl-1,
 nrhs=B.Cols, ldA=max(1,A.Rows), ldB=max(1,B.Rows), offsetA=0, offsetB=0)

 PURPOSE

 Solves linear equations
  A*X = B,   if trans is PNoTrans
  A^T*X = B, if trans is PTrans 
  A^H*X = B, if trans is PConjTrans

 On entry, A and ipiv contain the LU factorization of an n by n
 band matrix A as computed by Getrf() or Gbsv().  On exit B is
 replaced by the solution X.

 ARGUMENTS
  A         float or complex matrix
  B         float or complex  matrix.  Must have the same type as A.
  ipiv      int vector
  kl        nonnegative integer

 OPTIONS
  trans     PNoTrans, PTrans or PConjTrans
  n         nonnegative integer.  If negative, the default value is used.
  ku        nonnegative integer.  If negative, the default value is used.
  nrhs      nonnegative integer.  If negative, the default value is used.
  ldA       positive integer, ldA >= 2*kl+ku+1. If zero, the  default value is used.
  ldB       positive integer, ldB >= max(1,n). If zero, the default value is used.
  offsetA   nonnegative integer
  offsetB   nonnegative integer;
 */
func Gbtrs(A, B matrix.Matrix, ipiv []int32, KL int, opts ...linalg.Opt) error {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return err
	}
	ind := linalg.GetIndexOpts(opts...)
	//err = lapack_check(ind, fgbtrs, A, B, ipiv, pars)
	ind.Kl = KL
	if ind.Kl < 0 {
		return errors.New("invalid kl")
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
	if ind.LDb == 0 {
		ind.LDb = max(1, B.Rows())
	}
	if ind.OffsetB < 0 {
		return errors.New("offsetB")
	}
	sizeB := B.NumElements()
	if sizeB < ind.OffsetB+(ind.Nrhs-1)*ind.LDb + ind.N {
		return errors.New("sizeB")
	}
	if ipiv != nil && len(ipiv) < ind.N {
		return errors.New("size ipiv")
	}

	info := -1
	switch A.(type) {
	case *matrix.FloatMatrix:
		Aa := A.FloatArray()
		Ba := B.FloatArray()
		trans := linalg.ParamString(pars.Trans)
		info = dgbtrs(trans, ind.N, ind.Kl, ind.Ku, ind.Nrhs,
			Aa[ind.OffsetA:], ind.LDa, ipiv, Ba[ind.OffsetB:], ind.LDb)
	case *matrix.ComplexMatrix:
		return errors.New("Gbtrs for complex not yet implemented")
	}
	if info != 0 {
		return errors.New("gbtrs call error")
	}
	return nil
}



// Local Variables:
// tab-width: 4
// End:
