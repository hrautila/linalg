
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package lapack

// #cgo LDFLAGS: -L/usr/lib/libblas -L/usr/lib/lapack -llapack -lblas
// #include <stdlib.h>
// #include "lapack.h"
import "C"
import "unsafe"


//int ilaenv_(int  *ispec, char **name, char **opts, int *n1, int *n2, int *n3, int *n4);
func ilaenv(ispec int, name []string, opts []string, n1, n2, n3, n4 int) int {
	return 0
}

// void dgbsv_(int *n, int *kl, int *ku, int *nrhs,
//		double *AB, int *ldab, int *ipiv, double *b, int *ldb, int *info);
func dgbsv(n, kl, ku, nrhs int, A []float64, LDa int, ipiv []int32, B []float64, LDb int) int {
	var info int = 0
	C.dgbsv_((*C.int)(unsafe.Pointer(&n)), (*C.int)(unsafe.Pointer(&kl)),
		(*C.int)(unsafe.Pointer(&ku)), (*C.int)(unsafe.Pointer(&nrhs)),
		(*C.double)(unsafe.Pointer(&A[0])), (*C.int)(unsafe.Pointer(&LDa)),
		(*C.int)(unsafe.Pointer(&ipiv[0])), (*C.double)(unsafe.Pointer(&B[0])),
		(*C.int)(unsafe.Pointer(&LDb)), (*C.int)(unsafe.Pointer(&info)))
	return info
}

// void dgbtrf_(int *m, int *n, int *kl, int *ku,
//		double *AB, int *ldab, int *ipiv, int *info);
func dgbtrf(m, n, kl, ku int, AB []float64, ldab int, ipiv []int32) int {
	var info int = 0
	C.dgbtrf_((*C.int)(unsafe.Pointer(&m)),(*C.int)(unsafe.Pointer(&n)),
		(*C.int)(unsafe.Pointer(&kl)),(*C.int)(unsafe.Pointer(&ku)),
		(*C.double)(unsafe.Pointer(&AB[0])), (*C.int)(unsafe.Pointer(&ldab)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void dgbtrs_(char *trans, int *n, int *kl, int *ku, int *nrhs,
//		double *AB, int *ldab, int *ipiv, double *B, int *ldB, int *info);
//
func dgbtrs(trans string, n, kl, ku, nrhs int, A []float64, lda int, ipiv []int32, B []float64, ldb int) int {
	var info int = 0
	ctrans := C.CString(trans)
	defer C.free(unsafe.Pointer(ctrans))

	C.dgbtrs_(ctrans, (*C.int)(unsafe.Pointer(&n)), (*C.int)(unsafe.Pointer(&kl)),
		(*C.int)(unsafe.Pointer(&ku)),(*C.int)(unsafe.Pointer(&nrhs)),
		(*C.double)(unsafe.Pointer(&A[0])), (*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&ipiv[0])), (*C.double)(unsafe.Pointer(&B[0])),
		(*C.int)(unsafe.Pointer(&ldb)), (*C.int)(unsafe.Pointer(&info)))

	return info
}

// void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
func dgetrf(M, N int, A []float64, lda int, ipiv []int32) int {
	var info int = 0
	C.dgetrf_((*C.int)(unsafe.Pointer(&M)), (*C.int)(unsafe.Pointer(&N)),
		(*C.double)(unsafe.Pointer(&A[0])), (*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&ipiv[0])), (*C.int)(unsafe.Pointer(&info)))
	return info
}

// dgetrs_(char *trans, int *n, int *nrhs, double *A, int *lda,
//    int *ipiv, double *B, int *ldb, int *info);
func dgetrs(trans string, N, Nrhs int, A []float64, lda int, ipiv []int32,
	B []float64, ldb int) int {
	ctrans := C.CString(trans)
	defer C.free(unsafe.Pointer(ctrans))
	var info int = 0
	C.dgetrs_(ctrans, (*C.int)(unsafe.Pointer(&N)), (*C.int)(unsafe.Pointer(&Nrhs)),
		(*C.double)(unsafe.Pointer(&A[0])), (*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(*C.double)(unsafe.Pointer(&B[0])), (*C.int)(unsafe.Pointer(&ldb)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// dgetri_(int *n, double *A, int *lda, int *ipiv, double *work, int *lwork, int *info);
func dgetri(N int, A []float64, lda int, ipiv []int32) int {
	var info int = 0
	var lwork int = -1
	var work float64 
	// pre-calculate work buffer size
	C.dgetri_((*C.int)(unsafe.Pointer(&N)),	nil, (*C.int)(unsafe.Pointer(&lda)), nil,
		(*C.double)(unsafe.Pointer(&work)), (*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))

	// allocate work area
	lwork = int(work)
	wbuf := make([]float64, lwork)

	C.dgetri_((*C.int)(unsafe.Pointer(&N)),	(*C.double)(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)), (*C.int)(unsafe.Pointer(&ipiv[0])),
		(*C.double)(unsafe.Pointer(&wbuf[0])), (*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// dgesv_(int *n, int *nrhs, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
func dgesv(N, Nrhs int, A []float64, lda int, ipiv []int32,	B []float64, ldb int) int {
	var info int = 0
	C.dgesv_((*C.int)(unsafe.Pointer(&N)), (*C.int)(unsafe.Pointer(&Nrhs)),
		(*C.double)(unsafe.Pointer(&A[0])), (*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(*C.double)(unsafe.Pointer(&B[0])), (*C.int)(unsafe.Pointer(&ldb)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void dgttrf_(int *n, double *dl, double *d, double *du, double *du2, int *ipiv, int *info);
func dgttrf(N int, DL, D, DU, DU2 []float64, ipiv []int32) int {
	var info int = 0
	C.dgttrf_((*C.int)(unsafe.Pointer(&N)), (*C.double)(unsafe.Pointer(&DL[0])),
		(*C.double)(unsafe.Pointer(&D[0])),(*C.double)(unsafe.Pointer(&DU[0])),
		(*C.double)(unsafe.Pointer(&DU2[0])),(*C.int)(unsafe.Pointer(&ipiv[0])),		
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void dgttrs_(char *trans, int *n, int *nrhs, double *dl, double *d,
//		double *du, double *du2, int *ipiv, double *B, int *ldB, int *info);
func dgttrs(trans string, N, Nrhs int, DL, D, DU, DU2 []float64,
	ipiv []int32, B []float64, ldb int) int {
	var info int = 0
	ctrans := C.CString(trans)
	defer C.free(unsafe.Pointer(ctrans))
	C.dgttrs_(ctrans, (*C.int)(unsafe.Pointer(&N)),(*C.int)(unsafe.Pointer(&Nrhs)),
		(*C.double)(unsafe.Pointer(&DL[0])), (*C.double)(unsafe.Pointer(&D[0])),
		(*C.double)(unsafe.Pointer(&DU[0])), (*C.double)(unsafe.Pointer(&DU2[0])),
		(*C.int)(unsafe.Pointer(&ipiv[0])),	(*C.double)(unsafe.Pointer(&B[0])),
		(*C.int)(unsafe.Pointer(&ldb)),(*C.int)(unsafe.Pointer(&info)))		
	return info
}

// void dgtsv_(int *n, int *nrhs, double *DL, double *D,
//		double *DU, double *B, int *ldB, int *info);

// void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);
func dpotrf(uplo string, N int, A []float64, lda int) int {
	var info int
	cuplo := C.CString(uplo)
	defer C.free(unsafe.Pointer(cuplo))

	C.dpotrf_(cuplo,
		(*C.int)(unsafe.Pointer(&N)),
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void dpotrs_(char *uplo, int *n, int *nrhs, double *A,
//		int *lda, double *B, int *ldb, int *info);
func dpotrs(uplo string, N, Nrhs int, A []float64, lda int, B []float64, ldb int) int {
	var info int = 0
	cuplo := C.CString(uplo)
	defer C.free(unsafe.Pointer(cuplo))
	C.dpotrs_(cuplo,
		(*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&Nrhs)),
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.double)(unsafe.Pointer(&B[0])),
		(*C.int)(unsafe.Pointer(&ldb)),
		(*C.int)(unsafe.Pointer(&info)))
	return info

}

// void dpotri_(char *uplo, int *n, double *A, int *lda, int *info);
func dpotri(uplo string, N int, A []float64, lda int) int {
	var info int = 0
	cuplo := C.CString(uplo)
	defer C.free(unsafe.Pointer(cuplo))
	C.dpotri_(cuplo,
		(*C.int)(unsafe.Pointer(&N)),
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&info)))
	return info

}

// void dposv_(char *uplo, int *n, int *nrhs, double *A, int *lda,
//		double *B, int *ldb, int *info);
func dposv(uplo string, N, Nrhs int, A []float64, lda int, B []float64, ldb int) int {
	var info int = 0
	cuplo := C.CString(uplo)
	defer C.free(unsafe.Pointer(cuplo))
	C.dposv_(cuplo,
		(*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&Nrhs)),
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.double)(unsafe.Pointer(&B[0])),
		(*C.int)(unsafe.Pointer(&ldb)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void dpbtrf_(char *uplo, int *n, int *kd, double *AB, int *ldab, int *info);

// void dpbtrs_(char *uplo, int *n, int *kd, int *nrhs, double *AB,
//		int *ldab, double *B, int *ldb, int *info);

// void dpbsv_(char *uplo, int *n, int *kd, int *nrhs, double *A,
//		int *lda, double *B, int *ldb, int *info);

// void dpttrf_(int *n, double *d, double *e, int *info);

// void dpttrs_(int *n, int *nrhs, double *d, double *e, double *B, int *ldB, int *info);

// void dptsv_(int *n, int *nrhs, double *d, double *e, double *B, int *ldB, int *info);


// void dsytrf_(char *uplo, int *n, double *A, int *lda, int *ipiv,
//		double *work, int *lwork, int *info);
func dsytrf(uplo string, N int, A []float64, lda int, ipiv []int32) int {
	var info int = 0
	var lwork int = -1
	var work float64 
	cuplo := C.CString(uplo)
	defer C.free(unsafe.Pointer(cuplo))

	// pre-calculate work buffer size
	C.dsytrf_(cuplo, (*C.int)(unsafe.Pointer(&N)),
		nil, (*C.int)(unsafe.Pointer(&lda)), nil,
		(*C.double)(unsafe.Pointer(&work)), (*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))

	// allocate work area
	lwork = int(work)
	wbuf := make([]float64, lwork)

	C.dsytrf_(cuplo, (*C.int)(unsafe.Pointer(&N)),
		(*C.double)(unsafe.Pointer(&A[0])), (*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(*C.double)(unsafe.Pointer(&wbuf[0])), (*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}


// void dsytrs_(char *uplo, int *n, int *nrhs, double *A, int *lda,
//		int *ipiv, double *B, int *ldb, int *info);
func dsytrs(uplo string, N, Nrhs int, A []float64, lda int, ipiv []int32, B []float64, ldb int) int {

	var info int = 0
	cuplo := C.CString(uplo)
	defer C.free(unsafe.Pointer(cuplo))

	C.dsytrs_(cuplo, (*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&Nrhs)),
		(*C.double)(unsafe.Pointer(&A[0])), (*C.int)(unsafe.Pointer(&lda)),
		(*C.int)(unsafe.Pointer(&ipiv[0])),
		(*C.double)(unsafe.Pointer(&B[0])), (*C.int)(unsafe.Pointer(&ldb)),
		(*C.int)(unsafe.Pointer(&info)))
	return info

}


// void dsytri_(char *uplo, int *n, double *A, int *lda, int *ipiv,
//		double *work, int *info);

// void dsysv_(char *uplo, int *n, int *nrhs, double *A, int *lda,
//		int *ipiv, double *B, int *ldb, double *work, int *lwork, int *info);

// void dtrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs,
//		double  *A, int *lda, double *B, int *ldb, int *info);
func dtrtrs(uplo, trans, diag string, N, Nrhs int, A []float64, lda int, B []float64, ldb int) int {

	var info int = 0
	cuplo := C.CString(uplo)
	defer C.free(unsafe.Pointer(cuplo))
	ctrans := C.CString(trans)
	defer C.free(unsafe.Pointer(cuplo))
	cdiag := C.CString(diag)
	defer C.free(unsafe.Pointer(cdiag))

	C.dtrtrs_(cuplo, ctrans, cdiag,
		(*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&Nrhs)),
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.double)(unsafe.Pointer(&B[0])),
		(*C.int)(unsafe.Pointer(&ldb)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}


// void dtrtri_(char *uplo, char *diag, int *n, double  *a, int *lda, int *info);

// void dtbtrs_(char *uplo, char *trans, char *diag, int *n, int *kd,
//		int *nrhs, double *A, int *lda, double *B, int *ldb, int *info);

// void dgels_(char *trans, int *m, int *n, int *nrhs, double *A, int *lda,
//		double *B, int *ldb, double *work, int *lwork, int *info);

// void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau,
//		double *work, int *lwork, int *info);
func dgeqrf(M, N int, A []float64, lda int, tau []float64) int {
	var info int = 0
	var lwork int = -1
	var work float64 

	// calculate work buffer size
	C.dgeqrf_((*C.int)(unsafe.Pointer(&M)),
		(*C.int)(unsafe.Pointer(&N)),
		nil, 
		(*C.int)(unsafe.Pointer(&lda)),
		nil,
		(*C.double)(unsafe.Pointer(&work)),
		(*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))

		
	lwork = int(work)
	wbuf := make([]float64, lwork)
	C.dgeqrf_((*C.int)(unsafe.Pointer(&M)),
		(*C.int)(unsafe.Pointer(&N)),
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.double)(unsafe.Pointer(&tau[0])),
		(*C.double)(unsafe.Pointer(&wbuf[0])),
		(*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void dormqr_(char *side, char *trans, int *m, int *n, int *k,
//		double *a, int *lda, double *tau, double *c, int *ldc,
//		double *work, int *lwork, int *info);
func dormqr(side, trans string, M, N, K int, A []float64, lda int, tau, C []float64, ldc int) int {
	var info int = 0
	var lwork int = -1
	var work float64 

	cside := C.CString(side)
	defer C.free(unsafe.Pointer(cside))
	ctrans := C.CString(trans)
	defer C.free(unsafe.Pointer(ctrans))

	// calculate work buffer size
	C.dormqr_(cside, ctrans,
		(*C.int)(unsafe.Pointer(&M)),
		(*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&K)),
		nil, 
		(*C.int)(unsafe.Pointer(&lda)),
		nil,
		nil,
		(*C.int)(unsafe.Pointer(&ldc)),
		(*C.double)(unsafe.Pointer(&work)),
		(*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))
		
	lwork = int(work)
	wbuf := make([]float64, lwork)
	C.dormqr_(cside, ctrans,
		(*C.int)(unsafe.Pointer(&M)),
		(*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&K)),
		(*C.double)(unsafe.Pointer(&A[0])), 
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.double)(unsafe.Pointer(&tau[0])), 
		(*C.double)(unsafe.Pointer(&C[0])), 
		(*C.int)(unsafe.Pointer(&ldc)),
		(*C.double)(unsafe.Pointer(&wbuf[0])),
		(*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void dorgqr_(int *m, int *n, int *k, double *A, int *lda, double *tau,
//		double *work, int *lwork, int *info);

// void dorglq_(int *m, int *n, int *k, double *A, int *lda, double *tau,
//		double *work, int *lwork, int *info);

// void dgelqf_(int *m, int *n, double *a, int *lda, double *tau,
//		double *work, int *lwork, int *info);

// void dormlq_(char *side, char *trans, int *m, int *n, int *k, double *a,
//		int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info);

// void dgeqp3_(int *m, int *n, double *a, int *lda, int *jpvt, double *tau,
//		double *work, int *lwork, int *info);

// void dsyev_(char *jobz, char *uplo, int *n, double *A, int *lda, double *W,
//		double *work, int *lwork, int *info);

// void dsyevx_(char *jobz, char *range, char *uplo, int *n, double *A, int *lda,
//		double *vl, double *vu, int *il, int *iu, double *abstol, int *m,
//		double *W, double *Z, int *ldz, double *work, int *lwork, int *iwork,
//		int *ifail, int *info);

// void dsyevd_(char *jobz, char *uplo, int *n, double *A, int *ldA, double *W,
//		double *work, int *lwork, int *iwork, int *liwork, int *info);
func dsyevd(jobz, uplo string, N int, A []float64, lda int, W []float64) int {
	var info int = 0
	var lwork int = -1
	var liwork int = -1
	var iwork int32
	var work float64 

	cjobz := C.CString(jobz)
	defer C.free(unsafe.Pointer(cjobz))
	cuplo := C.CString(uplo)
	defer C.free(unsafe.Pointer(cuplo))
	
	// pre-calculate work buffer size
	C.dsyevd_(cjobz, cuplo, (*C.int)(unsafe.Pointer(&N)), nil,
		(*C.int)(unsafe.Pointer(&lda)), nil,
		(*C.double)(unsafe.Pointer(&work)), (*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&iwork)), (*C.int)(unsafe.Pointer(&liwork)),
		(*C.int)(unsafe.Pointer(&info)))

	// allocate work area
	lwork = int(work)
	wbuf := make([]float64, lwork)
	liwork = int(iwork)
	wibuf := make([]int32, liwork)

	C.dsyevd_(cjobz, cuplo, (*C.int)(unsafe.Pointer(&N)),
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)), (*C.double)(unsafe.Pointer(&W[0])),
		(*C.double)(unsafe.Pointer(&wbuf[0])), (*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&wibuf[0])), (*C.int)(unsafe.Pointer(&liwork)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// void dsyevr_(char *jobz, char *range, char *uplo, int *n, double *A, int *ldA,
//		double *vl, double *vu, int *il, int *iu, double *abstol, int *m, double *W,
//		double *Z, int *ldZ, int *isuppz, double *work, int *lwork, int *iwork,
//		int *liwork, int *info);
func dsyevr(jobz, srange, uplo string, N int, A []float64, lda int, vl, vu float64,
	il, iu int, M int, W, Z []float64, LDz int) int {

	var info int = 0
	var lwork int = -1
	var liwork int = -1
	var iwork int32
	var work float64 
	var abstol float64 = 0.0

	cjobz := C.CString(jobz)
	defer C.free(unsafe.Pointer(cjobz))
	cuplo := C.CString(uplo)
	defer C.free(unsafe.Pointer(cuplo))
	crange := C.CString(srange)
	defer C.free(unsafe.Pointer(crange))

	// pre-calculate work buffer size
	C.dsyevr_(cjobz, crange, cuplo, (*C.int)(unsafe.Pointer(&N)),
		nil, (*C.int)(unsafe.Pointer(&lda)), 
		(*C.double)(unsafe.Pointer(&vl)), (*C.double)(unsafe.Pointer(&vu)),
		(*C.int)(unsafe.Pointer(&il)), (*C.int)(unsafe.Pointer(&iu)), 
		(*C.double)(unsafe.Pointer(&abstol)), (*C.int)(unsafe.Pointer(&M)),
		nil, nil,
		(*C.int)(unsafe.Pointer(&LDz)), nil,
		(*C.double)(unsafe.Pointer(&work)), (*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&iwork)), (*C.int)(unsafe.Pointer(&liwork)),
		(*C.int)(unsafe.Pointer(&info)))

	// allocate work area
	lwork = int(work)
	wbuf := make([]float64, lwork)
	liwork = int(iwork)
	wibuf := make([]int32, liwork)

	var Zbuf, Wbuf *C.double
	if W != nil {
		Wbuf = (*C.double)(unsafe.Pointer(&W[0]))
	} else {
		Wbuf = (*C.double)(unsafe.Pointer(nil))
	}
	if Z != nil {
		Zbuf = (*C.double)(unsafe.Pointer(&Z[0]))
	} else {
		Zbuf = (*C.double)(unsafe.Pointer(nil))
	}
	
	C.dsyevr_(cjobz, crange, cuplo, (*C.int)(unsafe.Pointer(&N)),
		(*C.double)(unsafe.Pointer(&A[0])), (*C.int)(unsafe.Pointer(&lda)), 
		(*C.double)(unsafe.Pointer(&vl)), (*C.double)(unsafe.Pointer(&vu)),
		(*C.int)(unsafe.Pointer(&il)), (*C.int)(unsafe.Pointer(&iu)), 
		(*C.double)(unsafe.Pointer(&abstol)), (*C.int)(unsafe.Pointer(&M)),
		Wbuf, Zbuf,
		(*C.int)(unsafe.Pointer(&LDz)), nil,
		(*C.double)(unsafe.Pointer(&wbuf)), (*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&wibuf)), (*C.int)(unsafe.Pointer(&liwork)),
		(*C.int)(unsafe.Pointer(&info)))
	
	return info
}
	

// void dsygv_(int *itype, char *jobz, char *uplo, int *n, double *A, int *lda,
//		double *B, int *ldb, double *W, double *work, int *lwork,  int *info);

// void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *A, int *ldA,
//		double *S, double *U, int *ldU, double *Vt, int *ldVt, double *work,
//		int *lwork, int *info);
func dgesvd(jobu, jobvt string, M, N int, A []float64, lda int, S []float64, U []float64,
		ldu int, Vt []float64, ldvt int) int {

	var info int = 0
	var lwork int = -1
	var work float64 
	//var abstol float64 = 0.0

	cjobu := C.CString(jobu)
	defer C.free(unsafe.Pointer(cjobu))
	cjobvt := C.CString(jobvt)
	defer C.free(unsafe.Pointer(cjobvt))

	// pre-calculate work buffer size
	C.dgesvd_(cjobu, cjobvt, (*C.int)(unsafe.Pointer(&M)), (*C.int)(unsafe.Pointer(&N)),
		nil, (*C.int)(unsafe.Pointer(&lda)), 
		nil, nil, (*C.int)(unsafe.Pointer(&ldu)),
		nil,(*C.int)(unsafe.Pointer(&ldvt)),
		(*C.double)(unsafe.Pointer(&work)), (*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))

	// allocate work area
	lwork = int(work)
	wbuf := make([]float64, lwork)

	var Ubuf, Vtbuf *C.double
	if U != nil {
		Ubuf = (*C.double)(unsafe.Pointer(&U[0]))
	} else {
		Ubuf = (*C.double)(unsafe.Pointer(nil))
	}
	if Vt != nil {
		Vtbuf = (*C.double)(unsafe.Pointer(&Vt[0]))
	} else {
		Vtbuf = (*C.double)(unsafe.Pointer(nil))
	}

	C.dgesvd_(cjobu, cjobvt, (*C.int)(unsafe.Pointer(&M)), (*C.int)(unsafe.Pointer(&N)),
		(*C.double)(unsafe.Pointer(&A[0])), (*C.int)(unsafe.Pointer(&lda)), 
		(*C.double)(unsafe.Pointer(&S[0])), Ubuf, (*C.int)(unsafe.Pointer(&ldu)),
		Vtbuf,(*C.int)(unsafe.Pointer(&ldvt)),
		(*C.double)(unsafe.Pointer(&wbuf)), (*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}
// void dgesdd_(char *jobz, int *m, int *n, double *A, int *ldA, double *S,
//		double *U, int *ldU, double *Vt, int *ldVt, double *work, int *lwork,
//		int *iwork, int *info);


// void dgees_(char *jobvs, char *sort, void *select, int *n, double *A, int *ldA,
//		int *sdim, double *wr, double *wi, double *vs, int *ldvs, double *work,
//		int *lwork, int *bwork, int *info);

// void dgges_(char *jobvsl, char *jobvsr, char *sort, void *delctg, int *n,
//		double *A, int *ldA, double *B, int *ldB, int *sdim, double *alphar,
//		double *alphai, double *beta, double *vsl, int *ldvsl, double *vsr,
//		int *ldvsr, double *work, int *lwork, int *bwork, int *info);



// Local Variables:
// tab-width: 4
// End:


