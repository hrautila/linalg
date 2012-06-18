
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
 LDL^T factorization of a real or complex symmetric matrix.

 Sytrf(A, ipiv, uplo=PLower, n=A.Rows, ldA=max(1,A.Rows))

 PURPOSE
 Computes the LDL^T factorization of a real or complex symmetric
 n by n matrix  A.  On exit, A and ipiv contain the details of the
 factorization.

 ARGUMENTS
  A         float or complex matrix
  ipiv      int vector of length at least n

 OPTIONS
  uplo      PLower or PUpper
  n         nonnegative integer.  If negative, the default value is  used.
  ldA       positive integer.  ldA >= max(1,n).  If zero, the default
            value is used.
  offsetA   nonnegative integer;

*/
func Sytrf(A matrix.Matrix, ipiv []int32, opts ...linalg.Option) error {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return err
	}
	ind := linalg.GetIndexOpts(opts...)
	if ind.N < 0 {
		ind.N = A.Rows()
		if ind.N != A.Cols() {
			return errors.New("A not square")
		}
	}
	if ind.N == 0  {
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
	sizeA := A.NumElements()
	if sizeA < ind.OffsetA+(ind.N-1)*ind.LDa+ind.N {
		return errors.New("sizeA")
	}
	if ipiv != nil && len(ipiv) < ind.N {
		return errors.New("size ipiv")
	}
	info := -1
	switch A.(type) {
	case *matrix.FloatMatrix:
		Aa := A.FloatArray()
		uplo := linalg.ParamString(pars.Uplo)
		info = dsytrf(uplo, ind.N, Aa[ind.OffsetA:], ind.LDa, ipiv)
	case *matrix.ComplexMatrix:
	}
	if info != 0 {
		return errors.New("sytrf failed")
	}
	return nil
}


// Local Variables:
// tab-width: 4
// End:
