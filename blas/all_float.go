
// Copyright (c) Harri Rautila, 2012

// This file is part of github.com/hrautila/linalg package. 
// It is free software, distributed under the terms of GNU Lesser General Public 
// License Version 3, or any later version. See the COPYING tile included in this archive.

package blas

import (
	"github.com/hrautila/linalg"
	"github.com/hrautila/matrix"
	//"errors"
	"math"
)

// See function Nrm2.
func Nrm2Float(X *matrix.FloatMatrix, opts ...linalg.Option) (v float64) {
	v = math.NaN()
	ind := linalg.GetIndexOpts(opts...)
	err := check_level1_func(ind, fnrm2, X, nil)
	if err != nil {
		return
	}
	if ind.Nx == 0 {
		v = 0.0
		return
	}
	Xa := X.FloatArray()
	v = dnrm2(ind.Nx, Xa[ind.OffsetX:], ind.IncX)
	return
}

// See function Asum.
func AsumFloat(X *matrix.FloatMatrix, opts ...linalg.Option) (v float64) {
	v = math.NaN()
	ind := linalg.GetIndexOpts(opts...)
	err := check_level1_func(ind, fasum, X, nil)
	if err != nil {
		return
	}
	if ind.Nx == 0 {
		v = 0.0
		return
	}
	Xa := X.FloatArray()
	v = dasum(ind.Nx, Xa[ind.OffsetX:], ind.IncX)
	return
}


// See functin Dot.
func DotFloat(X, Y *matrix.FloatMatrix, opts ...linalg.Option) (v float64) {
	v = math.NaN()
	ind := linalg.GetIndexOpts(opts...)
	err := check_level1_func(ind, fdot, X, Y)
	if err != nil {
		return
	}
	if ind.Nx == 0 {
		v = 0.0
		return 
	}
	Xa := X.FloatArray()
	Ya := Y.FloatArray()
	v = ddot(ind.Nx, Xa[ind.OffsetX:], ind.IncX, Ya[ind.OffsetY:], ind.IncY)
	return
}

// See function Swap.
func SwapFloat(X, Y *matrix.FloatMatrix, opts ...linalg.Option) (err error) {
	ind := linalg.GetIndexOpts(opts...)
	err = check_level1_func(ind, fswap, X, Y)
	if err != nil {
		return
	}
	if ind.Nx == 0 {
		return
	}
	Xa := X.FloatArray()
	Ya := Y.FloatArray()
	dswap(ind.Nx, Xa[ind.OffsetX:], ind.IncX, Ya[ind.OffsetY:], ind.IncY)
	return
}

// See function Copy.
func CopyFloat(X, Y *matrix.FloatMatrix, opts ...linalg.Option) (err error) {
	ind := linalg.GetIndexOpts(opts...)
	err = check_level1_func(ind, fcopy, X, Y)
	if err != nil {
		return
	}
	if ind.Nx == 0 {
		return
	}
	Xa := X.FloatArray()
	Ya := Y.FloatArray()
	dcopy(ind.Nx, Xa[ind.OffsetX:], ind.IncX, Ya[ind.OffsetY:], ind.IncY)
	return
}

// Calculate for column vector X := alpha * X. Contents of X is set to new value.
// Valid options: offset, inc, N.

// See function Scal.
func ScalFloat(X *matrix.FloatMatrix, alpha float64, opts ...linalg.Option) (err error) {
	ind := linalg.GetIndexOpts(opts...)
	err = check_level1_func(ind, fscal, X, nil)
	if err != nil {
		return
	}
	if ind.Nx == 0 {
		return
	}
	Xa := X.FloatArray()
	dscal(ind.Nx, alpha, Xa[ind.OffsetX:], ind.IncX)
	return
}

// See function Axpy.
func AxpyFloat(X, Y *matrix.FloatMatrix, alpha float64, opts ...linalg.Option) (err error) {
	ind := linalg.GetIndexOpts(opts...)
	err = check_level1_func(ind, faxpy, X, Y)
	if err != nil {
		return
	}
	if ind.Nx == 0 {
		return
	}
	Xa := X.FloatArray()
	Ya := Y.FloatArray()
	daxpy(ind.Nx, alpha, Xa[ind.OffsetX:], ind.IncX, Ya[ind.OffsetY:], ind.IncY)
	return
}


// ---------------------------------------------------------------------------------
// BLAS LEVEL 2 
// ---------------------------------------------------------------------------------

// See function Gemv.
func GemvFloat(A, X, Y *matrix.FloatMatrix, alpha, beta float64, opts ...linalg.Option) (err error) {

	var params *linalg.Parameters
	params, err = linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level2_func(ind, fgemv, X, Y, A, params)
	if err != nil {
		return
	}
	if ind.M == 0 && params.Trans == linalg.PNoTrans {
		return
	}
	if ind.N == 0 && (params.Trans == linalg.PTrans || params.Trans == linalg.PConjTrans) {
		return
	}
	Xa := X.FloatArray()
	Ya := Y.FloatArray()
	Aa := A.FloatArray()
	if params.Trans == linalg.PNoTrans && ind.N == 0 {
		dscal(ind.M, beta, Ya[ind.OffsetY:], ind.IncY)
	} else if params.Trans == linalg.PTrans && ind.M == 0 {
		dscal(ind.N, beta, Ya[ind.OffsetY:], ind.IncY)
	} else {
		trans := linalg.ParamString(params.Trans)
		dgemv(trans, ind.M, ind.N, alpha, Aa[ind.OffsetA:],
			ind.LDa, Xa[ind.OffsetX:], ind.IncX, beta, Ya[ind.OffsetY:], ind.IncY)
	}
	return
}

/*
 Matrix-vector product with a general banded matrix.

 PURPOSE:
 If trans is 'NoTrans', computes Y := alpha*A*X + beta*Y
 If trans is 'Trans', computes Y := alpha*A^T*X + beta*Y.
 The matrix A is m by n with upper bandwidth ku and lower bandwidth kl.
 Returns immediately if n=0 and trans is 'Trans', or if m=0 and trans is 'N'.
 Computes y := beta*y if n=0, m>0, and trans is 'NoTrans', or if m=0, n>0,
 and trans is 'Trans'

 ARGUMENTS
 X         float n*1 matrix.  
 Y         float m*1 matrix
 A         float m*n matrix.
 alpha     number (float). 
 beta      number (float).
 trans     NoTrans or Trans

 OPTIONS:
 m         nonnegative integer, default A.rows
 kl        nonnegative integer
 n         nonnegative integer.  If negative, the default value is used.
 ku        nonnegative integer.  If negative, the default value is used.
 ldA       positive integer.  ldA >= kl+ku+1. If zero, the default value is used.
 incx      nonzero integer, default =1
 incy      nonzero integer, default =1
 offsetA   nonnegative integer, default =0
 offsetx   nonnegative integer, default =0
 offsety   nonnegative integer, default =0

*/

// See function Gbmv.
func GbmvFloat(A, X, Y *matrix.FloatMatrix, alpha, beta float64, opts ...linalg.Option) (err error) {

	var params *linalg.Parameters
	params, err = linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level2_func(ind, fgbmv, X, Y, A, params)
	if err != nil {
		return
	}
	if ind.M == 0 && ind.N == 0 {
		return
	}

	Xa := X.FloatArray()
	Ya := Y.FloatArray()
	Aa := A.FloatArray()
	if params.Trans == linalg.PNoTrans && ind.N == 0 {
		dscal(ind.M, beta, Ya[ind.OffsetY:], ind.IncY)
	} else if params.Trans == linalg.PTrans && ind.M == 0 {
		dscal(ind.N, beta, Ya[ind.OffsetY:], ind.IncY)
	} else {
		trans := linalg.ParamString(params.Trans)
		dgbmv(trans, ind.M, ind.N, ind.Kl, ind.Ku,
			alpha, Aa[ind.OffsetA:], ind.LDa, Xa[ind.OffsetX:], ind.IncX,
			beta, Ya[ind.OffsetY:], ind.IncY)
	}
	return
}

/*
 Matrix-vector product with a real symmetric matrix.

 PURPOSE
 Computes y := alpha*A*x + beta*y with A real symmetric of order n.

 ARGUMENTS
 A         float n*n matrix
 x         float n*1 matrix
 y         float n*1 matrix
 uplo      Lower or Upper
 alpha     real number (float)
 beta      real number (float)

 OPTIONS:
 n         integer.  If negative, the default value is used.
           If the default value is used, we require that
           A.size[0]=A.size[1].
 ldA       nonnegative integer.  ldA >= max(1,n).
           If zero, the default value is used.
 incx      nonzero integer
 incy      nonzero integer
 offsetA   nonnegative integer
 offsetx   nonnegative integer
 offsety   nonnegative integer
*/

// See function Symv.
func SymvFloat(A, X, Y *matrix.FloatMatrix, alpha, beta float64, opts ...linalg.Option) (err error) {

	var params *linalg.Parameters
	params, err = linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level2_func(ind, fsymv, X, Y, A, params)
	if err != nil {
		return
	}
	if ind.N == 0 {
		return
	}
	Xa := X.FloatArray()
	Ya := Y.FloatArray()
	Aa := A.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	dsymv(uplo, ind.N, alpha, Aa[ind.OffsetA:], ind.LDa, Xa[ind.OffsetX:], ind.IncX,
		beta, Ya[ind.OffsetY:], ind.IncY)
	return
}

/*
 Matrix-vector product with a real symmetric band matrix.

 PURPOSE
 Computes y := alpha*A*x + beta*y with A real symmetric and 
 banded of order n and with bandwidth k.

 ARGUMENTS
 A         float n*n matrix
 x         float n*1 matrix
 y         float n*1 matrix
 uplo      'L' or 'U'
 alpha     real number
 beta      real number

 OPTIONS:
 n         integer.  If negative, the default value is used.
 k         integer.  If negative, the default value is used.
           The default value is k = max(0,A.size[0]-1).
 ldA       nonnegative integer.  ldA >= k+1.
           If zero, the default vaule is used.
 incx      nonzero integer
 incy      nonzero integer
 offsetA   nonnegative integer
 offsetx   nonnegative integer
 offsety   nonnegative integer

*/

// See function Sbmv.
func SbmvFloat(A, X, Y *matrix.FloatMatrix, alpha, beta float64, opts ...linalg.Option) (err error) {

	var params *linalg.Parameters
	params, err = linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level2_func(ind, fsbmv, X, Y, A, params)
	if err != nil {
		return
	}
	if ind.N == 0 {
		return
	}
	Xa := X.FloatArray()
	Ya := Y.FloatArray()
	Aa := A.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	dsbmv(uplo, ind.N, ind.K, alpha, Aa[ind.OffsetA:], ind.LDa, Xa[ind.OffsetX:],
		ind.IncX, beta, Ya[ind.OffsetY:], ind.IncY)
	return
}

/*
 Matrix-vector product with a triangular matrix.
 trmv(A, x, uplo='L', trans='N', diag='N', n=A.size[0],
      ldA=max(1,A.size[0]), incx=1, offsetA=0, offsetx=0)

 PURPOSE
 If trans is 'N', computes x := A*x.
 If trans is 'T', computes x := A^T*x.
 A is triangular of order n.

 ARGUMENTS
 A         'd' or 'z' matrix
 x         'd' or 'z' matrix.  Must have the same type as A.
 uplo      'L' or 'U'
 trans     'N' or 'T
 diag      'N' or 'U'
 n         integer.  If negative, the default value is used.
           If the default value is used, we require that
           A.size[0] = A.size[1].
 ldA       nonnegative integer.  ldA >= max(1,n).
           If zero the default value is used.
 incx      nonzero integer, default=1
 offsetA   nonnegative integer, default=0
 offsetx   nonnegative integer, default=0

*/

// See function Trmv.
func TrmvFloat(A, X *matrix.FloatMatrix, opts ...linalg.Option) (err error) {

	var params *linalg.Parameters
	params, err = linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level2_func(ind, ftrmv, X, nil, A, params)
	if err != nil {
		return
	}
	if ind.N == 0 {
		return
	}
	Xa := X.FloatArray()
	Aa := A.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	trans := linalg.ParamString(params.Trans)
	diag := linalg.ParamString(params.Diag)
	dtrmv(uplo, trans, diag, ind.N,
		Aa[ind.OffsetA:], ind.LDa, Xa[ind.OffsetX:], ind.IncX)
	return
}

/*
 Matrix-vector product with a triangular band matrix.
 tbmv(A, x, uplo='L', trans='N', diag='N', n=A.size[1],
      k=max(0,A.size[0]-1), ldA=A.size[0], incx=1, offsetA=0,
      offsetx=0)
 PURPOSE
 If trans is 'N', computes x := A*x.
 If trans is 'T', computes x := A^T*x.
 If trans is 'C', computes x := A^H*x.
 A is banded triangular of order n and with bandwith k.
 ARGUMENTS
 A         float m*n matrix
 x         float n*1 matrix.
 uplo      'Lower' or 'Upper'
 trans     'NoTrans', 'Trans'
 diag      'NoUnit' or 'Unit'
 n         nonnegative integer.  If negative, the default value is used.
 k         nonnegative integer.  If negative, the default value is used.
 ldA       nonnegative integer.  lda >= 1+k.
           If zero the default value is used.
 incx      nonzero integer
 offsetA   nonnegative integer
 offsetx   nonnegative integer

*/

// See function Tbmv.
func TbmvFloat(A, X *matrix.FloatMatrix, opts ...linalg.Option) (err error) {

	var params *linalg.Parameters
	params, err = linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level2_func(ind, ftbmv, X, nil, A, params)
	if err != nil {
		return
	}
	if ind.N == 0 {
		return
	}
	Xa := X.FloatArray()
	Aa := A.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	trans := linalg.ParamString(params.Trans)
	diag := linalg.ParamString(params.Diag)
	dtbmv(uplo, trans, diag, ind.N, ind.K,
		Aa[ind.OffsetA:], ind.LDa, Xa[ind.OffsetX:], ind.IncX)
	return
}

/*
 Solution of a triangular set of equations with one righthand side.
 trsv(A, x, uplo='L', trans='N', diag='N', n=A.size[0],
      ldA=max(1,A.size[0]), incx=1, offsetA=0, offsetx=0)
 PURPOSE
 If trans is 'NoTrans', computes x := A^{-1}*x.
 If trans is 'Trans', computes x := A^{-T}*x.
 A is triangular of order n.  The code does not verify whether A is nonsingular.

 ARGUMENTS
 A         float matrix
 x         float matrix.
 uplo      'Lower' or 'Upper'
 trans     'NoTrans', 'Trans' 
 diag      'NoUnit' or 'Unit'

 OPTIONS
 n         integer.  If negative, the default value is used.
           If the default value is used, we require that A.rows = A.cols.
 ldA       nonnegative integer.  ldA >= max(1,n). If zero, the default value is used.
 incx      nonzero integer
 offsetA   nonnegative integer
 offsetx   nonnegative integer;
*/

// See function Trsv.
func TrsvFloat(A, X *matrix.FloatMatrix, opts ...linalg.Option) (err error) {

	var params *linalg.Parameters
	params, err = linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level2_func(ind, ftrsv, X, nil, A, params)
	if err != nil {
		return
	}
	if ind.N == 0 {
		return
	}
	Xa := X.FloatArray()
	Aa := A.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	trans := linalg.ParamString(params.Trans)
	diag := linalg.ParamString(params.Diag)
	dtrsv(uplo, trans, diag, ind.N,
		Aa[ind.OffsetA:], ind.LDa, Xa[ind.OffsetX:], ind.IncX)
	return
}

/*
 Solution of a triangular and banded set of equations.
 tbsv(A, x, uplo='L', trans='N', diag='N', n=A.size[1],
      k=max(0,A.size[0]-1), ldA=A.size[0], incx=1, offsetA=0,
      offsetx=0)
 PURPOSE
 If trans is 'N', computes x := A^{-1}*x.
 If trans is 'T', computes x := A^{-T}*x.
 A is banded triangular of order n and with bandwidth k.

 ARGUMENTS
 A         float matrix
 x         float matrix.
 uplo      'L' or 'U'
 trans     'N', 'T' or 'C'
 diag      'N' or 'U'

 OPTIONS
 n         nonnegative integer.  If negative, the default value is used.
 k         nonnegative integer.  If negative, the default value is used.
 ldA       nonnegative integer.  ldA >= 1+k.
           If zero the default value is used.
 incx      nonzero integer
 offsetA   nonnegative integer
 offsetx   nonnegative integer;
*/

// See function Tbsv.
func TbsvFloat(A, X *matrix.FloatMatrix, opts ...linalg.Option) (err error) {

	var params *linalg.Parameters
	params, err = linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level2_func(ind, ftbsv, X, nil, A, params)
	if err != nil {
		return
	}
	if ind.N == 0 {
		return
	}
	Xa := X.FloatArray()
	Aa := A.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	trans := linalg.ParamString(params.Trans)
	diag := linalg.ParamString(params.Diag)
	dtbsv(uplo, trans, diag, ind.N, ind.K,
		Aa[ind.OffsetA:], ind.LDa, Xa[ind.OffsetX:], ind.IncX)
	return
}

/*
 General rank-1 update.
 ger(x, y, A, alpha=1.0, m=A.size[0], n=A.size[1], incx=1,
     incy=1, ldA=max(1,A.size[0]), offsetx=0, offsety=0,
     offsetA=0)
 PURPOSE
 Computes A := A + alpha*x*y^H with A m by n, real or complex.

 ARGUMENTS
 x         float matrix.
 y         float matrix.
 A         float matrix.
 alpha     number (float).

 OPTIONS
 m         integer.  If negative, the default value is used.
 n         integer.  If negative, the default value is used.
 incx      nonzero integer
 incy      nonzero integer
 ldA       nonnegative integer.  ldA >= max(1,m).
           If zero, the default value is used.
 offsetx   nonnegative integer
 offsety   nonnegative integer
 offsetA   nonnegative integer;

*/

// See function Ger.
func GerFloat(X, Y, A *matrix.FloatMatrix, alpha float64, opts ...linalg.Option) (err error) {

	var params *linalg.Parameters
	params, err = linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level2_func(ind, fger, X, Y, A, params)
	if err != nil {
		return
	}
	if ind.N == 0 || ind.M == 0 {
		return
	}
	Xa := X.FloatArray()
	Ya := Y.FloatArray()
	Aa := A.FloatArray()
	dger(ind.M, ind.N,	alpha, Xa[ind.OffsetX:], ind.IncX,
		Ya[ind.OffsetY:], ind.IncY, Aa[ind.OffsetA:], ind.LDa)

	return
}

/*
 Symmetric rank-1 update.
 syr(x, A, uplo='L', alpha=1.0, n=A.size[0], incx=1,
     ldA=max(1,A.size[0]), offsetx=0, offsetA=0)
 PURPOSE
 Computes A := A + alpha*x*x^T with A real symmetric of order n.

 ARGUMENTS
 x         'd' matrix
 A         'd' matrix
 uplo      'L' or 'U'
 alpha     real number (int or float)
 n         integer.  If negative, the default value is used.
 incx      nonzero integer
 ldA       nonnegative integer.  ldA >= max(1,n).
           If zero, the default value is used.
 offsetx   nonnegative integer
 offsetA   nonnegative integer;
*/

// See function Syr.
func SyrFloat(X, A *matrix.FloatMatrix, alpha float64, opts ...linalg.Option) (err error) {

	var params *linalg.Parameters
	params, err = linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level2_func(ind, fsyr, X, nil, A, params)
	if err != nil {
		return
	}
	if ind.N == 0 {
		return
	}
	Xa := X.FloatArray()
	Aa := A.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	dsyr(uplo, ind.N, alpha, Xa[ind.OffsetX:], ind.IncX, Aa[ind.OffsetA:], ind.LDa)
	return
}

/*
 Symmetric rank-2 update.
 syr2(x, y, A, uplo='L', alpha=1.0, n=A.size[0], incx=1, incy=1,
     ldA=max(1,A.size[0]), offsetx=0, offsety=0, offsetA=0)
 PURPOSE
 Computes A := A + alpha*(x*y^T + y*x^T) with A real symmetric.
 ARGUMENTS
 x         'd' matrix
 y         'd' matrix
 A         'd' matrix
 uplo      'L' or 'U'
 alpha     real number (int or float)
 n         integer.  If negative, the default value is used.
 incx      nonzero integer
 incy      nonzero integer
 ldA       nonnegative integer.  ldA >= max(1,n).
           If zero the default value is used.
 offsetx   nonnegative integer
 offsety   nonnegative integer
 offsetA   nonnegative integer;
*/

// See function Syr2.
func Syr2Float(X, Y, A *matrix.FloatMatrix, alpha float64, opts ...linalg.Option) (err error) {

	var params *linalg.Parameters
	params, err = linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level2_func(ind, fsyr2, X, Y, A, params)
	if err != nil {
		return
	}
	if ind.N == 0 {
		return
	}
	Xa := X.FloatArray()
	Ya := X.FloatArray()
	Aa := A.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	dsyr2(uplo, ind.N, alpha, Xa[ind.OffsetX:], ind.IncX, Ya[ind.OffsetY:], ind.IncY,
		Aa[ind.OffsetA:], ind.LDa)
	return
}

// ---------------------------------------------------------------------------------
// BLAS LEVEL 3
// ---------------------------------------------------------------------------------

/*
 General matrix-matrix product.
 gemm(A, B, C, transA='N', transB='N', alpha=1.0, beta=0.0, 
      m=None, n=None, k=None, ldA=max(1,A.size[0]), 
      ldB=max(1,B.size[0]), ldC=max(1,C.size[0]), offsetA=0, 
      offsetB=0, offsetC=0) 
 PURPOSE
 Computes 
 C := alpha*A*B + beta*C     if transA = 'N' and transB = 'N'.
 C := alpha*A^T*B + beta*C   if transA = 'T' and transB = 'N'.
 C := alpha*A*B^T + beta*C   if transA = 'N' and transB = 'T'.
 C := alpha*A^T*B^T + beta*C if transA = 'T' and transB = 'T'.
 The number of rows of the matrix product is m.  The number of 
 columns is n.  The inner dimension is k.  If k=0, this reduces 
 to C := beta*C.

 ARGUMENTS
 A         float matrix, m*k
 B         float matrix, k*n
 C         float matrix, m*n
 alpha     number float
 beta      number float

 OPTIONS:
 transA    'NoTrans', 'Trans'
 transB    'NoTrans', 'Trans' or 'C'
 m         integer.  If negative, the default value is used. The default value is
           m = (transA == 'NoTrans') ? A.Rows : A.Cols.
 n         integer.  If negative, the default value is used. The default value is
           n = (transB == 'NoTrans') ? B.Cols : B.Rows.
 k         integer.  If negative, the default value is used. The default value is
           (transA == 'NoTrans') ? A.Cols : A.Rows, transA='N'.
           If the default value is used it should also be equal to
           (transB == 'N') ? B.Rows : B.Cols.
 ldA       nonnegative integer.  ldA >= max(1,(transA == 'NoTrans') ? m : k).
           If zero, the default value is used.
 ldB       nonnegative integer.  ldB >= max(1,(transB == 'NoTrans') ? k : n).
           If zero, the default value is used.
 ldC       nonnegative integer.  ldC >= max(1,m).
           If zero, the default value is used.
 offsetA   nonnegative integer
 offsetB   nonnegative integer
 offsetC   nonnegative integer;
*/

// See function Gemm.
func GemmFloat(A, B, C *matrix.FloatMatrix, alpha, beta float64, opts ...linalg.Option) (err error) {

	params, e := linalg.GetParameters(opts...)
	if e != nil {
		err = e
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level3_func(ind, fgemm, A, B, C, params)
	if err != nil {
		return
	}
	if ind.M == 0 || ind.N == 0 {
		return
	}
	Aa := A.FloatArray()
	Ba := B.FloatArray()
	Ca := C.FloatArray()
	transB := linalg.ParamString(params.TransB)
	transA := linalg.ParamString(params.TransA)
	//diag := linalg.ParamString(params.Diag)
	dgemm(transA, transB, ind.M, ind.N, ind.K, alpha,
		Aa[ind.OffsetA:], ind.LDa, Ba[ind.OffsetB:], ind.LDb, beta,
		Ca[ind.OffsetC:], ind.LDc)
	return
}

/*
 Matrix-matrix product where one matrix is symmetric.

 symm(A, B, C, side='L', uplo='L', alpha=1.0, beta=0.0, 
      m=B.size[0], n=B.size[1], ldA=max(1,A.size[0]), 
      ldB=max(1,B.size[0]), ldC=max(1,C.size[0]), offsetA=0, 
      offsetB=0, offsetC=0)
 PURPOSE
 If side is 'Left', computes C := alpha*A*B + beta*C.
 If side is 'Right', computes C := alpha*B*A + beta*C.
 C is m by n and A is real symmetric. 

 ARGUMENTS
 A         float matrix
 B         float matrix.
 C         float m*n matrix.
 side      'Left' or 'Right'
 uplo      'Lower' or 'Upper'
 alpha     number (float).
 beta      number (float). 
 m         integer.  If negative, the default value is used.
           If the default value is used and side = 'Left', then m
           must be equal to A.size[0] and A.size[1].
 n         integer.  If negative, the default value is used.
           If the default value is used and side = 'Right', then 
           must be equal to A.size[0] and A.size[1].
 ldA       nonnegative integer.
           ldA >= max(1, (side == 'Left') ? m : n).
		   If zero, the default value is used.
 ldB       nonnegative integer.
           ldB >= max(1, (side == 'Lelft') ? n : m).  If zero, the default value is used.
 ldC       nonnegative integer.  ldC >= max(1,m). If zero, the default value is used.
 offsetA   nonnegative integer
 offsetB   nonnegative integer"
 "offsetC   nonnegative integer";

*/

// See function Symm.
func SymmFloat(A, B, C *matrix.FloatMatrix, alpha, beta float64, opts ...linalg.Option) (err error) {

	params, e := linalg.GetParameters(opts...)
	if e != nil {
		err = e
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level3_func(ind, fsymm, A, B, C, params)
	if err != nil {
		return
	}
	if ind.M == 0 || ind.N == 0 {
		return
	}
	Aa := A.FloatArray()
	Ba := B.FloatArray()
	Ca := C.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	side := linalg.ParamString(params.Side)
	dsymm(side, uplo, ind.M, ind.N, alpha, Aa[ind.OffsetA:], ind.LDa,
		Ba[ind.OffsetB:], ind.LDb, beta, Ca[ind.OffsetC:], ind.LDc)

	return
}

/*
 Rank-k update of symmetric matrix.
 syrk(A, C, uplo='L', trans='N', alpha=1.0, beta=0.0, n=None, 
      k=None, ldA=max(1,A.size[0]), ldC=max(1,C.size[0]),
      offsetA=0, offsetB=0)
 PURPOSE   
 If trans is 'N', computes C := alpha*A*A^T + beta*C.
 If trans is 'T', computes C := alpha*A^T*A + beta*C.
 C is symmetric (real or complex) of order n. 
 The inner dimension of the matrix product is k.  If k=0 this is
 interpreted as C := beta*C.
 ARGUMENTS
 A         'd' or 'z' matrix
 C         'd' or 'z' matrix.  Must have the same type as A.
 uplo      'L' or 'U'
 trans     'N' or 'T'
 alpha     number (int, float or complex).  Complex alpha is only
           allowed if A is complex.
 beta      number (int, float or complex).  Complex beta is only
           allowed if A is complex.
 n         integer.  If negative, the default value is used.
           The default value is
           n = (trans == N) ? A.size[0] : A.size[1].
 k         integer.  If negative, the default value is used.
           The default value is
           k = (trans == 'N') ? A.size[1] : A.size[0].
 ldA       nonnegative integer.
           ldA >= max(1, (trans == 'N') ? n : k).  If zero,
           the default value is used.
 ldC       nonnegative integer.  ldC >= max(1,n).
           If zero, the default value is used.
 offsetA   nonnegative integer
 offsetC   nonnegative integer;
*/

// See function Syrk.
func SyrkFloat(A, C *matrix.FloatMatrix, alpha, beta float64, opts ...linalg.Option) (err error) {

	params, e := linalg.GetParameters(opts...)
	if e != nil {
		err = e
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level3_func(ind, fsyrk, A, nil, C, params)
	if e != nil || err != nil {
		return
	}
	if ind.N == 0 {
		return
	}
	Aa := A.FloatArray()
	Ca := C.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	trans := linalg.ParamString(params.Trans)
	//diag := linalg.ParamString(params.Diag)
	dsyrk(uplo, trans, ind.N, ind.K, alpha, Aa[ind.OffsetA:], ind.LDa, beta,
		Ca[ind.OffsetC:], ind.LDc)
		
	return
}

/*
 Rank-2k update of symmetric matrix.
 syr2k(A, B, C, uplo='L', trans='N', alpha=1.0, beta=0.0, n=None,
       k=None, ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), 
       ldC=max(1,C.size[0])), offsetA=0, offsetB=0, offsetC=0)
 PURPOSE
 If trans is 'N', computes C := alpha*(A*B^T + B*A^T) + beta*C.
 If trans is 'T', computes C := alpha*(A^T*B + B^T*A) + beta*C.
 C is symmetric (real or complex) of order n.
 The inner dimension of the matrix product is k.  If k=0 this is
 interpreted as C := beta*C.
 ARGUMENTS
 A         'd' or 'z' matrix
 B         'd' or 'z' matrix.  Must have the same type as A.
 C         'd' or 'z' matrix.  Must have the same type as A.
 uplo      'L' or 'U'
 trans     'N', 'T' or 'C' ('C' is only allowed when in the real
n           case and means the same as 'T')
 alpha     number (int, float or complex).  Complex alpha is only
           allowed if A is complex.
 beta      number (int, float or complex).  Complex beta is only
           allowed if A is complex.
 n         integer.  If negative, the default value is used.
           The default value is
           n = (trans == 'N') ? A.size[0] : A.size[1].
           If the default value is used, it should be equal to
           (trans == 'N') ? B.size[0] : B.size[1].
 k         integer.  If negative, the default value is used.
           The default value is
           k = (trans == 'N') ? A.size[1] : A.size[0].
           If the default value is used, it should be equal to
           (trans == 'N') ? B.size[1] : B.size[0].
 ldA       nonnegative integer.
           ldA >= max(1, (trans=='N') ? n : k).
           If zero, the default value is used.
 ldB       nonnegative integer.
           ldB >= max(1, (trans=='N') ? n : k).
           If zero, the default value is used.
 ldC       nonnegative integer.  ldC >= max(1,n).
           If zero, the default value is used.
 offsetA   nonnegative integer
 offsetB   nonnegative integer
 offsetC   nonnegative integer
 
 */

// See function Syrk2.
func Syr2kFloat(A, B, C *matrix.FloatMatrix, alpha, beta float64, opts ...linalg.Option) (err error) {

	params, e := linalg.GetParameters(opts...)
	if e != nil {
		err = e
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level3_func(ind, fsyr2k, A, B, C, params)
	if err != nil {
		return
	}
	if ind.N == 0 {
		return
	}
	Aa := A.FloatArray()
	Ba := B.FloatArray()
	Ca := C.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	trans := linalg.ParamString(params.Trans)
	//diag := linalg.ParamString(params.Diag)
	dsyr2k(uplo, trans,	ind.N, ind.K, alpha, Aa[ind.OffsetA:], ind.LDa,
		Ba[ind.OffsetB:], ind.LDb, beta, Ca[ind.OffsetC:], ind.LDc)
	return
}

/*
 Matrix-matrix product where one matrix is triangular.
 trmm(A, B, side='L', uplo='L', transA='N', diag='N', alpha=1.0,
      m=None, n=None, ldA=max(1,A.size[0]), ldB=max(1,B.size[0]),
      offsetA=0, offsetB=0)
 PURPOSE
 Computes
 B := alpha*A*B   if transA is 'N' and side = 'L'.
 B := alpha*B*A   if transA is 'N' and side = 'R'.
 B := alpha*A^T*B if transA is 'T' and side = 'L'.
 B := alpha*B*A^T if transA is 'T' and side = 'R'.
 B := alpha*A^H*B if transA is 'C' and side = 'L'.
 B := alpha*B*A^H if transA is 'C' and side = 'R'.
 B is m by n and A is triangular.
 ARGUMENTS
 A         'd' or 'z' matrix
 B         'd' or 'z' matrix.  Must have the same type as A.
 side      'L' or 'R'
 uplo      'L' or 'U'
 transA    'N' or 'T'
 diag      'N' or 'U'
 alpha     number (int, float or complex).  Complex alpha is only
           allowed if A is complex.
 m         integer.  If negative, the default value is used.
           The default value is
           m = (side == 'L') ? A.size[0] : B.size[0].
           If the default value is used and side is 'L', m must
           be equal to A.size[1].
 n         integer.  If negative, the default value is used.
           The default value is
           n = (side == 'L') ? B.size[1] : A.size[0].
           If the default value is used and side is 'R', n must
           be equal to A.size[1].
 ldA       nonnegative integer.
           ldA >= max(1, (side == 'L') ? m : n).
           If zero, the default value is used. 
 ldB       nonnegative integer.  ldB >= max(1,m).
           If zero, the default value is used.
 offsetA   nonnegative integer
 offsetB   nonnegative integer
 */

// See function Trmm.
func TrmmFloat(A, B *matrix.FloatMatrix, alpha float64, opts ...linalg.Option) (err error) {

	params, e := linalg.GetParameters(opts...)
	if e != nil {
		err = e
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level3_func(ind, ftrmm, A, B, nil, params)
	if err != nil {
		return
	}
	if ind.M == 0 || ind.N == 0 {
		return
	}
	Aa := A.FloatArray()
	Ba := B.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	transA := linalg.ParamString(params.TransA)
	side := linalg.ParamString(params.Side)
	diag := linalg.ParamString(params.Diag)
	dtrmm(side, uplo, transA, diag,	ind.M, ind.N, alpha,
		Aa[ind.OffsetA:], ind.LDa, Ba[ind.OffsetB:], ind.LDb)
	return
}

/*
 Solution of a triangular system of equations with multiple 
 righthand sides.
 trsm(A, B, side='L', uplo='L', transA='N', diag='N', alpha=1.0,
      m=None, n=None, ldA=max(1,A.size[0]), ldB=max(1,B.size[0]),
      offsetA=0, offsetB=0)
 PURPOSE
 Computes
 B := alpha*A^{-1}*B if transA is 'N' and side = 'L'.
 B := alpha*B*A^{-1} if transA is 'N' and side = 'R'.
 B := alpha*A^{-T}*B if transA is 'T' and side = 'L'.
 B := alpha*B*A^{-T} if transA is 'T' and side = 'R'.
 B := alpha*A^{-H}*B if transA is 'C' and side = 'L'.
 B := alpha*B*A^{-H} if transA is 'C' and side = 'R'.
 B is m by n and A is triangular.  The code does not verify 
 whether A is nonsingular.
 ARGUMENTS
 A         'd' or 'z' matrix
 B         'd' or 'z' matrix.  Must have the same type as A.
 side      'Left' or 'Right'
 uplo      'Lower' or 'Upper'
 transA    'NoTrans' or 'Trans'
 diag      'NoUnit' or 'Unit'
 alpha     number (int, float or complex).  Complex alpha is only
           allowed if A is complex.
 m         integer.  If negative, the default value is used.
           The default value is
           m = (side == 'Left') ? A.rows : B.rows.
           If the default value is used and side is 'Left', m must
           be equal to A.cols.
 n         integer.  If negative, the default value is used.
           The default value is
           n = (side == 'Left') ? B.cols : A.rows.
           If the default value is used and side is 'Right', n must
           be equal to A.cols.
 ldA       nonnegative integer.
           ldA >= max(1, (side == 'Left') ? m : n).
           If zero, the default value is used.
 ldB       nonnegative integer.  ldB >= max(1,m).
           If zero, the default value is used.
 offsetA   nonnegative integer
 offsetB   nonnegative integer
 */

// See function Trsm.
func TrsmFloat(A, B *matrix.FloatMatrix, alpha float64, opts ...linalg.Option) (err error) {

	params, e := linalg.GetParameters(opts...)
	if e != nil {
		err = e
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	err = check_level3_func(ind, ftrsm, A, B, nil, params)
	if err != nil {
		return
	}
	if ind.N == 0 || ind.M == 0 {
		return
	}
	Aa := A.FloatArray()
	Ba := B.FloatArray()
	uplo := linalg.ParamString(params.Uplo)
	transA := linalg.ParamString(params.TransA)
	side := linalg.ParamString(params.Side)
	diag := linalg.ParamString(params.Diag)
	dtrsm(side, uplo, transA, diag,	ind.M, ind.N, alpha,
		Aa[ind.OffsetA:], ind.LDa, Ba[ind.OffsetB:], ind.LDb)
	return
}


// Local Variables:
// tab-width: 4
// End:
