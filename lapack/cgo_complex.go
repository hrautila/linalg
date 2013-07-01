// Copyright (c) Harri Rautila, 2012

// This file is part of github.com/hrautila/linalg/lapack package.
// It is free software, distributed under the terms of GNU Lesser General Public 
// License Version 3, or any later version. See the COPYING tile included in this archive.

package lapack

// #cgo LDFLAGS: -L/usr/lib/libblas -L/usr/lib/lapack -llapack -lblas
// #include <stdlib.h>
// #include "lapack.h"
import "C"
import "unsafe"

// void zlacpy_(char *uplo, int *m, int *n, complex *A, int *lda, complex *B, int *ldb);

// void zgetrf_(int *m, int *n, complex *A, int *lda, int *ipiv, int *info);
func zgetrf(M, N int, A []complex128, lda int, ipiv []int32) int {
	var info int = 0
	C.zgetrf_((*C.int)(unsafe.Pointer(&M)),
		(*C.int)(unsafe.Pointer(&N)),
		(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void zgetrs_(char *trans, int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, int *info);
func zgetrs(trans string, N, Nrhs int, A []complex128, lda int, ipiv []int32, B []complex128, ldb int) int {
	ctrans := C.CString(trans)
	defer C.free(unsafe.Pointer(ctrans))
	var info int = 0
	C.zgetrs_(ctrans,
		(*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&Nrhs)),
		(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(unsafe.Pointer(&B[0])),
		(*C.int)(unsafe.Pointer(&ldb)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void zgetri_(int *n, complex *A, int *lda, int *ipiv, complex *work, int *lwork, int *info);
func zgetri(N int, A []complex128, lda int, ipiv []int32) int {
	var info int = 0
	var lwork int = -1
	var work complex128
	// pre-calculate work buffer size
	C.zgetri_((*C.int)(unsafe.Pointer(&N)), nil, (*C.int)(unsafe.Pointer(&lda)), nil,
		(unsafe.Pointer(&work)), (*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))

	// allocate work area
	lwork = int(real(work))
	wbuf := make([]complex128, lwork)

	C.zgetri_((*C.int)(unsafe.Pointer(&N)),
		(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(unsafe.Pointer(&wbuf[0])),
		(*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void zgesv_(int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, int *info);
func zgesv(N, Nrhs int, A []complex128, lda int, ipiv []int32, B []complex128, ldb int) int {
	var info int = 0
	C.zgesv_((*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&Nrhs)),
		(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(unsafe.Pointer(&B[0])),
		(*C.int)(unsafe.Pointer(&ldb)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void zgbtrf_(int *m, int *n, int *kl, int *ku, complex *AB, int *ldab, int *ipiv, int *info);
func zgbtrf(m, n, kl, ku int, AB []complex128, ldab int, ipiv []int32) int {
	var info int = 0
	C.zgbtrf_((*C.int)(unsafe.Pointer(&m)),
		(*C.int)(unsafe.Pointer(&n)),
		(*C.int)(unsafe.Pointer(&kl)),
		(*C.int)(unsafe.Pointer(&ku)),
		unsafe.Pointer(&AB[0]),
		(*C.int)(unsafe.Pointer(&ldab)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void zgbtrs_(char *trans, int *n, int *kl, int *ku, int *nrhs, complex *AB, int *ldab, int *ipiv, complex *B, int *ldB, int *info);
func zgbtrs(trans string, n, kl, ku, nrhs int, A []complex128, lda int, ipiv []int32, B []complex128, ldb int) int {
	var info int = 0
	ctrans := C.CString(trans)
	defer C.free(unsafe.Pointer(ctrans))

	C.zgbtrs_(ctrans,
		(*C.int)(unsafe.Pointer(&n)),
		(*C.int)(unsafe.Pointer(&kl)),
		(*C.int)(unsafe.Pointer(&ku)),
		(*C.int)(unsafe.Pointer(&nrhs)),
		(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(unsafe.Pointer(&B[0])),
		(*C.int)(unsafe.Pointer(&ldb)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void zgbsv_(int *n, int *kl, int *ku, int *nrhs, complex *ab, int *ldab, int *ipiv, complex *b, int *ldb, int *info);
func zgbsv(n, kl, ku, nrhs int, A []complex128, LDa int, ipiv []int32, B []complex128, LDb int) int {
	var info int = 0
	C.zgbsv_((*C.int)(unsafe.Pointer(&n)),
		(*C.int)(unsafe.Pointer(&kl)),
		(*C.int)(unsafe.Pointer(&ku)),
		(*C.int)(unsafe.Pointer(&nrhs)),
		(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&LDa)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(unsafe.Pointer(&B[0])),
		(*C.int)(unsafe.Pointer(&LDb)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void zgttrf_(int *n, complex *dl, complex *d, complex *du, complex *du2, int *ipiv, int *info);
func zgttrf(N int, DL, D, DU, DU2 []complex128, ipiv []int32) int {
	var info int = 0
	C.zgttrf_((*C.int)(unsafe.Pointer(&N)),
		(unsafe.Pointer(&DL[0])),
		(unsafe.Pointer(&D[0])),
		(unsafe.Pointer(&DU[0])),
		(unsafe.Pointer(&DU2[0])),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void zgttrs_(char *trans, int *n, int *nrhs, complex *dl, complex *d, complex *du, complex *du2, int *ipiv, complex *B, int *ldB, int *info);
func zgttrs(trans string, N, Nrhs int, DL, D, DU, DU2 []complex128, ipiv []int32, B []complex128, ldb int) int {
	var info int = 0
	ctrans := C.CString(trans)
	defer C.free(unsafe.Pointer(ctrans))
	C.zgttrs_(ctrans,
		(*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&Nrhs)),
		(unsafe.Pointer(&DL[0])),
		(unsafe.Pointer(&D[0])),
		(unsafe.Pointer(&DU[0])),
		(unsafe.Pointer(&DU2[0])),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(unsafe.Pointer(&B[0])),
		(*C.int)(unsafe.Pointer(&ldb)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void zgtsv_(int *n, int *nrhs, complex *dl, complex *d, complex *du, complex *B, int *ldB, int *info);
// void zpotrf_(char *uplo, int *n, complex *A, int *lda, int *info);

// void zpotrs_(char *uplo, int *n, int *nrhs, complex *A, int *lda, complex *B, int *ldb, int *info);
// void zpotri_(char *uplo, int *n, complex *A, int *lda, int *info);
// void zposv_(char *uplo, int *n, int *nrhs, complex *A, int *lda, complex *B, int *ldb, int *info);
// void zpbtrf_(char *uplo, int *n, int *kd, complex *AB, int *ldab, int *info);
// void zpbtrs_(char *uplo, int *n, int *kd, int *nrhs, complex *AB, int *ldab, complex *B, int *ldb, int *info);
// void zpbsv_(char *uplo, int *n, int *kd, int *nrhs, complex *A, int *lda, complex *B, int *ldb, int *info);
// void zpttrf_(int *n, double *d, complex *e, int *info);
// void zpttrs_(char *uplo, int *n, int *nrhs, double *d, complex *e, complex *B, int *ldB, int *info);
// void zptsv_(int *n, int *nrhs, double *d, complex *e, complex *B, int *ldB, int *info);
// void zsytrf_(char *uplo, int *n, complex *A, int *lda, int *ipiv, complex *work, int *lwork, int *info);
// void zhetrf_(char *uplo, int *n, complex *A, int *lda, int *ipiv, complex *work, int *lwork, int *info);
// void zsytrs_(char *uplo, int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, int *info);
// void zhetrs_(char *uplo, int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, int *info);
// void zsytri_(char *uplo, int *n, complex *A, int *lda, int *ipiv, complex *work, int *info);
// void zhetri_(char *uplo, int *n, complex *A, int *lda, int *ipiv, complex *work, int *info);
// void zsysv_(char *uplo, int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, complex *work, int *lwork, int *info);
// void zhesv_(char *uplo, int *n, int *nrhs, complex *A, int *lda, int *ipiv, complex *B, int *ldb, complex *work, int *lwork, int *info);
// void ztrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, complex  *a, int *lda, complex *b, int *ldb, int *info);
// void ztrtri_(char *uplo, char *diag, int *n, complex  *a, int *lda, int *info);
// void ztbtrs_(char *uplo, char *trans, char *diag, int *n, int *kd, int *nrhs, complex *ab, int *ldab, complex *b, int *ldb, int *info);
// void zgels_(char *trans, int *m, int *n, int *nrhs, complex *a, int *lda, complex *b, int *ldb, complex *work, int *lwork, int *info);
// void zgeqrf_(int *m, int *n, complex *a, int *lda, complex *tau, complex *work, int *lwork, int *info);
// void zunmqr_(char *side, char *trans, int *m, int *n, int *k, complex *a, int *lda, complex *tau, complex *c, int *ldc, complex *work, int *lwork, int *info);
// void zungqr_(int *m, int *n, int *k, complex *A, int *lda, complex *tau, complex *work, int *lwork, int *info);
// void zunglq_(int *m, int *n, int *k, complex *A, int *lda, complex *tau, complex *work, int *lwork, int *info);
// void zgelqf_(int *m, int *n, complex *a, int *lda, complex *tau, complex *work, int *lwork, int *info);
// void zunmlq_(char *side, char *trans, int *m, int *n, int *k, complex *a, int *lda, complex *tau, complex *c, int *ldc, complex *work, int *lwork, int *info);
// void zgeqp3_(int *m, int *n, complex *a, int *lda, int *jpvt, complex *tau, complex *work, int *lwork, double *rwork, int *info);
// void zheev_(char *jobz, char *uplo, int *n, complex *A, int *lda, double *W, complex *work, int *lwork, double *rwork, int *info);
// void zheevx_(char *jobz, char *range, char *uplo, int *n, complex *A, int *lda, double *vl, double *vu, int *il, int *iu, double *abstol, int *m, double *W, complex *Z, int *ldz, complex *work, int *lwork, double *rwork, int *iwork, int *ifail, int *info);
// void zheevd_(char *jobz, char *uplo, int *n, complex *A, int *ldA, double *W, complex *work, int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *info);
// void zheevr_(char *jobz, char *range, char *uplo, int *n, complex *A, int *ldA, double *vl, double *vu, int *il, int *iu, double *abstol, int *m, double *W, complex *Z, int *ldZ, int *isuppz, complex *work, int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *info);
// void zhegv_(int *itype, char *jobz, char *uplo, int *n, complex *A, int *lda, complex *B, int *ldb, double *W, complex *work, int *lwork, double *rwork, int *info);
// void zgesvd_(char *jobu, char *jobvt, int *m, int *n, complex *A,
// void zgesdd_(char *jobz, int *m, int *n, complex *A, int *ldA, double *S, complex *U, int *ldU, complex *Vt, int *ldVt, complex *work, int *lwork, double *rwork, int *iwork, int *info);
// void zgees_(char *jobvs, char *sort, void *select, int *n, complex *A, int *ldA, int *sdim, complex *w, complex *vs, int *ldvs, complex *work, int *lwork, complex *rwork, int *bwork, int *info);
// void zgges_(char *jobvsl, char *jobvsr, char *sort, void *delctg, int *n, complex *A, int *ldA, complex *B, int *ldB, int *sdim, complex *alpha, complex *beta, complex *vsl, int *ldvsl, complex *vsr, int *ldvsr, complex *work, int *lwork, double *rwork, int *bwork, int *info);

// Local Variables:
// tab-width: 4
// End:
