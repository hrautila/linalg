
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
 Singular value decomposition of a real or complex matrix.

 Gesvd(A, S, jobu=PJobNo, jobvt=PJobNo, U=nil, Vt=nil, m=A.Rows,
 n=A.Cols, ldA=max(1,A.Rows), ldU=-1, ldVt=-1,
 offsetA=0, offsetS=0, offsetU=0, offsetVt=0)

 PURPOSE

 Computes singular values and, optionally, singular vectors of a 
 real or complex m by n matrix A.

 The argument jobu controls how many left singular vectors are
 computed: 

  PJobNo : no left singular vectors are computed.
  PJobAll: all left singular vectors are computed and returned as
           columns of U.
  PJobS  : the first min(m,n) left singular vectors are computed and
           returned as columns of U.
  PJobO  : the first min(m,n) left singular vectors are computed and
           returned as columns of A.
 
 The argument jobvt controls how many right singular vectors are
 computed:

  PJobNo : no right singular vectors are computed.
  PJobAll: all right singular vectors are computed and returned as
           rows of Vt.
  PJobS  : the first min(m,n) right singular vectors are computed and
           returned as rows of Vt.
  PJobO  : the first min(m,n) right singular vectors are computed and
           returned as rows of A.

 Note that the (conjugate) transposes of the right singular 
 vectors are returned in Vt or A.
 On exit (in all cases), the contents of A are destroyed.
 
 ARGUMENTS
  A         float or complex matrix
  S         float matrix of length at least min(m,n).  On exit, 
            contains the computed singular values in descending order.
  jobu      PJobNo, PJobAll, PJobS or PJobO
  jobvt     PJobNo, PJobAll, PJobS or PJobO
  U         float or complex matrix.  Must have the same type as A.
            Not referenced if jobu is PJobNo or PJobO.  If jobu is PJobAll,
            a matrix with at least m columns.   If jobu is PJobS, a
            matrix with at least min(m,n) columns.
            On exit (with jobu PJobAll or PJobS), the columns of U
            contain the computed left singular vectors.
  Vt        float or complex matrix.  Must have the same type as A.
            Not referenced if jobvt is PJobNo or PJobO.  If jobvt is 
            PJobAll or PJobS, a matrix with at least n columns.
            On exit (with jobvt PJobAll or PJobS), the rows of Vt
            contain the computed right singular vectors, or, in
            the complex case, their complex conjugates.
  m         integer.  If negative, the default value is used.
  n         integer.  If negative, the default value is used.
  ldA       nonnegative integer.  ldA >= max(1,m).
            If zero, the default value is used.
  ldU       nonnegative integer.
            ldU >= 1        if jobu is PJobNo or PJobO
            ldU >= max(1,m) if jobu is PJobAll or PJobS.
            The default value is max(1,U.Rows) if jobu is PJobAll 
            or PJobS, and 1 otherwise.
            If zero, the default value is used.
  ldVt      nonnegative integer.
            ldVt >= 1 if jobvt is PJobNo or PJobO.
            ldVt >= max(1,n) if jobvt is PJobAll.  
            ldVt >= max(1,min(m,n)) if ldVt is PJobS.
            The default value is max(1,Vt.Rows) if jobvt is PJobAll
            or PJobS, and 1 otherwise.
            If zero, the default value is used.
  offsetA   nonnegative integer
  offsetS   nonnegative integer
  offsetU   nonnegative integer
  offsetVt  nonnegative integer

 */
func Gesvd(A, S, U, Vt matrix.Matrix, opts ...linalg.Option) error {
	if ! matrix.EqualTypes(A, S, U, Vt) {
		return errors.New("Gesvd: arguments not same type")
	}
	switch A.(type) {
	case *matrix.FloatMatrix:
		Am := A.(*matrix.FloatMatrix)
		Sm := S.(*matrix.FloatMatrix)
		Um := U.(*matrix.FloatMatrix)
		Vm := Vt.(*matrix.FloatMatrix)
		return GesvdFloat(Am, Sm, Um, Vm, opts...)
	case *matrix.ComplexMatrix:
		Am := A.(*matrix.ComplexMatrix)
		Sm := S.(*matrix.ComplexMatrix)
		Um := U.(*matrix.ComplexMatrix)
		Vm := Vt.(*matrix.ComplexMatrix)
		return GesvdComplex(Am, Sm, Um, Vm, opts...)
	}
	return errors.New("Gesvd: unknown parameter types")
}

func GesvdFloat(A, S, U, Vt *matrix.FloatMatrix, opts ...linalg.Option) error {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return err
	}
	ind := linalg.GetIndexOpts(opts...)
	err = checkGesvd(ind, pars, A, S, U, Vt)
	if err != nil {
		return err
	}
	if ind.M == 0 || ind.N == 0 {
		return nil
	}
	Aa := A.FloatArray()
	Sa := S.FloatArray()
	var Ua, Va []float64
	Ua = nil; Va = nil
	if U != nil {
		Ua = U.FloatArray()[ind.OffsetU:]
	}
	if Vt != nil {
		Va = Vt.FloatArray()[ind.OffsetVt:]
	}
	info := dgesvd(linalg.ParamString(pars.Jobu), linalg.ParamString(pars.Jobvt),
		ind.M, ind.N, Aa[ind.OffsetA:], ind.LDa, Sa[ind.OffsetS:], Ua, ind.LDu, Va, ind.LDvt)
	if info != 0 {
		return errors.New("GesvdFloat not implemented yet")
	}
	return nil
}

func GesvdComplex(A, S, U, Vt *matrix.ComplexMatrix, opts ...linalg.Option) error {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return err
	}
	ind := linalg.GetIndexOpts(opts...)
	err = checkGesvd(ind, pars, A, S, U, Vt)
	if err != nil {
		return err
	}
	if ind.M == 0 || ind.N == 0 {
		return nil
	}
	return errors.New("GesvdComplex not implemented yet")
}

func checkGesvd(ind *linalg.IndexOpts, pars *linalg.Parameters, A, S, U, Vt matrix.Matrix) error {
	if ind.M < 0 {
		ind.M = A.Rows()
	}
	if ind.N < 0 {
		ind.N = A.Cols()
	}
	if ind.M == 0 || ind.N == 0 {
		return nil
	}
	if pars.Jobu == linalg.PJobO && pars.Jobvt == linalg.PJobO {
		return errors.New("Gesvd: jobu and jobvt cannot both have value PJobO")
	}
	if pars.Jobu == linalg.PJobAll || pars.Jobu == linalg.PJobS {
		if U == nil {
			return errors.New("Gesvd: missing matrix U")
		}
		if ind.LDu == 0 {
			ind.LDu = max(1, U.Rows())
		}
		if ind.LDu < max(1, ind.M) {
			return errors.New("Gesvd: ldU")
		}
	} else {
		if ind.LDu == 0 { ind.LDu = 1 }
		if ind.LDu < 1 {
			return errors.New("Gesvd: ldU")
		}
	}
	if pars.Jobvt == linalg.PJobAll || pars.Jobvt == linalg.PJobS {
		if Vt == nil {
			return errors.New("Gesvd: missing matrix Vt")
		}
		if ind.LDvt == 0 {
			ind.LDvt = max(1, Vt.Rows())
		}
		if pars.Jobvt == linalg.PJobAll && ind.LDvt < max(1, ind.N)  {
			return errors.New("Gesvd: ldVt")
		} else if pars.Jobvt != linalg.PJobAll && ind.LDvt < max(1, min(ind.M, ind.N))  {
			return errors.New("Gesvd: ldVt")
		}
	} else {
		if ind.LDvt == 0 { ind.LDvt = 1 }
		if ind.LDvt < 1 {
			return errors.New("Gesvd: ldVt")
		}
	}
	if ind.OffsetA < 0 {
		return errors.New("Gesvd: offsetA")
	}
	sizeA := A.NumElements()
	if sizeA < ind.OffsetA + (ind.N-1)*ind.LDa + ind.M {
		return errors.New("Gesvd: sizeA")
	}

	if ind.OffsetS < 0 {
		return errors.New("Gesvd: offsetS")
	}
	sizeS := S.NumElements()
	if sizeS < ind.OffsetS + min(ind.M, ind.N) {
		return errors.New("Gesvd: sizeA")
	}

	if ind.OffsetU < 0 {
		return errors.New("Gesvd: OffsetU")
	}
	sizeU := U.NumElements()
	if pars.Jobu == linalg.PJobAll && sizeU < ind.LDu*(ind.M-1) {
		return errors.New("Gesvd: sizeU")
	} else if pars.Jobu == linalg.PJobS && sizeU < ind.LDu*(min(ind.M,ind.N)-1) {
		return errors.New("Gesvd: sizeU")
	}

	if ind.OffsetVt < 0 {
		return errors.New("Gesvd: OffsetVt")
	}
	sizeVt := Vt.NumElements()
	if pars.Jobvt == linalg.PJobAll && sizeVt <  ind.N {
		return errors.New("Gesvd: sizeVt")
	} else if pars.Jobvt == linalg.PJobS && sizeVt < min(ind.M, ind.N) {
		return errors.New("Gesvd: sizeVt")
	}
	return nil
}


// Local Variables:
// tab-width: 4
// End:
