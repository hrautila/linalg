// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg/lapack package.
// It is free software, distributed under the terms of GNU Lesser General Public 
// License Version 3, or any later version. See the COPYING tile included in this archive.

package lapack

// #cgo LDFLAGS: -L/usr/lib/libblas -L/usr/lib/lapack -llapack -lblas
// #include <stdlib.h>
// #include "lapackutil.h"
import "C"
import "unsafe"

func dlarf(side string, M, N int, V []float64, incv int, tau, C []float64, ldc int) {
	var work []float64

	cside := C.CString(side)
	defer C.free(unsafe.Pointer(cside))
	if side[0] == 'L' {
		work = make([]float64, N, N)
	} else {
		work = make([]float64, M, M)
	}

	C.dlarf_(cside,
		(*C.int)(unsafe.Pointer(&M)),
		(*C.int)(unsafe.Pointer(&N)),
		(*C.double)(unsafe.Pointer(&V[0])),
		(*C.int)(unsafe.Pointer(&incv)),
		(*C.double)(unsafe.Pointer(&tau[0])),
		(*C.double)(unsafe.Pointer(&C[0])),
		(*C.int)(unsafe.Pointer(&ldc)),
		(*C.double)(unsafe.Pointer(&work[0])))
	return
}

func dlarfb(side, trans, direct, storev string, M, N, K int, V []float64, ldv int, T []float64, ldt int, C []float64, ldc int) {
	var work []float64
	var ldwork int

	cside := C.CString(side)
	defer C.free(unsafe.Pointer(cside))
	ctrans := C.CString(trans)
	defer C.free(unsafe.Pointer(ctrans))
	cdirect := C.CString(direct)
	defer C.free(unsafe.Pointer(cdirect))
	cstorev := C.CString(storev)
	defer C.free(unsafe.Pointer(cstorev))

	if side[0] == 'L' {
		ldwork = N * K
	} else {
		ldwork = M * K
	}
	work = make([]float64, ldwork, ldwork)

	C.dlarfb_(cside, ctrans, cdirect, cstorev,
		(*C.int)(unsafe.Pointer(&M)),
		(*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&K)),
		(*C.double)(unsafe.Pointer(&V[0])),
		(*C.int)(unsafe.Pointer(&ldv)),
		(*C.double)(unsafe.Pointer(&T[0])),
		(*C.int)(unsafe.Pointer(&ldt)),
		(*C.double)(unsafe.Pointer(&C[0])),
		(*C.int)(unsafe.Pointer(&ldc)),
		(*C.double)(unsafe.Pointer(&work[0])),
		(*C.int)(unsafe.Pointer(&ldwork)))
	return
}

func dlarfx(side string, M, N int, V []float64, tau, C []float64, ldc int) {
	var work []float64

	cside := C.CString(side)
	defer C.free(unsafe.Pointer(cside))
	if side[0] == 'L' {
		work = make([]float64, N, N)
	} else {
		work = make([]float64, M, M)
	}

	C.dlarfx_(cside,
		(*C.int)(unsafe.Pointer(&M)),
		(*C.int)(unsafe.Pointer(&N)),
		(*C.double)(unsafe.Pointer(&V[0])),
		(*C.double)(unsafe.Pointer(&tau[0])),
		(*C.double)(unsafe.Pointer(&C[0])),
		(*C.int)(unsafe.Pointer(&ldc)),
		(*C.double)(unsafe.Pointer(&work[0])))
	return
}

func dlarfg(N int, alpha, X []float64, incx int, tau []float64) {

	C.dlarfg_(
		(*C.int)(unsafe.Pointer(&N)),
		(*C.double)(unsafe.Pointer(&alpha[0])),
		(*C.double)(unsafe.Pointer(&X[0])),
		(*C.int)(unsafe.Pointer(&incx)),
		(*C.double)(unsafe.Pointer(&tau[0])))
	return
}

func dlarft(direct, storev string, N, K int, V []float64, ldv int, tau, T []float64, ldt int) {
	cstore := C.CString(storev)
	defer C.free(unsafe.Pointer(cstore))
	cdirect := C.CString(direct)
	defer C.free(unsafe.Pointer(cdirect))

	C.dlarft_(cdirect, cstore,
		(*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&K)),
		(*C.double)(unsafe.Pointer(&V[0])),
		(*C.int)(unsafe.Pointer(&ldv)),
		(*C.double)(unsafe.Pointer(&tau[0])),
		(*C.double)(unsafe.Pointer(&T[0])),
		(*C.int)(unsafe.Pointer(&ldt)))
	return
}

func dorgqr(M, N, K int, A []float64, lda int, tau []float64) int {
	var info int = 0
	var lwork int = -1
	var work float64

	// pre-calculate work buffer size
	C.dorgqr_(
		(*C.int)(unsafe.Pointer(&M)),
		(*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&K)),
		nil,
		(*C.int)(unsafe.Pointer(&lda)),
		nil,
		(*C.double)(unsafe.Pointer(&work)),
		(*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))

	// allocate work area
	lwork = int(work)
	wbuf := make([]float64, lwork)

	C.dorgqr_(
		(*C.int)(unsafe.Pointer(&M)),
		(*C.int)(unsafe.Pointer(&N)),
		(*C.int)(unsafe.Pointer(&K)),
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.int)(unsafe.Pointer(&lda)),
		(*C.double)(unsafe.Pointer(&tau[0])),
		(*C.double)(unsafe.Pointer(&wbuf[0])),
		(*C.int)(unsafe.Pointer(&lwork)),
		(*C.int)(unsafe.Pointer(&info)))
	return info
}

// Local Variables:
// tab-width: 4
// End:
