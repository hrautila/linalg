
// Copyright (c) Harri Rautila, 2012

// This file is part of github.com/hrautila/linalg/blas package. 
// It is free software, distributed under the terms of GNU Lesser General Public 
// License Version 3, or any later version. See the COPYING tile included in this archive.


package blas

import (
	"github.com/hrautila/linalg"
	"github.com/hrautila/matrix"
	"errors"
	//"fmt"
)

type funcNum int
const (
	fnrm2 = 1 + iota
	fasum
	fiamax
	fdot
	fswap
	fcopy
	fset
	faxpy
	faxpby
	fscal
	frotg
	frotmg
	frot
	frotm
	fgemv
	fgbmv
	fdtrm
	ftbmv
	ftpmv
	ftrsv
	ftbsv
	ftpsv
	ftrmv
	fsymv
	fsbmv
	fspmv
	fger
	fsyr
	fspr
	fsyr2
	fdspr2
	fgemm
	fsymm
	fsyrk
	fsyr2k
	ftrmm
	ftrsm
	)


func abs(val int) int {
	if val < 0 {
		return -val
	}
	return val
}

func max(a, b int) int {
	if a < b {
		return b
	}
	return a
}




func check_level1_func(ind *linalg.IndexOpts, fn funcNum, X, Y matrix.Matrix) error {

	nX, nY := 0, 0
	// this is adapted from cvxopt:blas.c python blas interface
	switch fn {
	case fnrm2, fasum, fiamax, fscal, fset:
		if ind.IncX <= 0 {
			return errors.New("incX illegal, <=0")
		}
		if ind.OffsetX < 0 {
			return errors.New("offsetX illegal, <0")
		}
		sizeX := X.NumElements()
		if sizeX >= ind.OffsetX + 1 {
			// calculate default size for N based on X size
			nX = 1 + (sizeX - ind.OffsetX - 1)/ind.IncX
		}
		if sizeX < ind.OffsetX + 1 + (ind.Nx-1)*abs(ind.IncX) {
			return errors.New("X size error")
		}
		if ind.Nx < 0 {
			ind.Nx = nX
		}
		
	case fdot, fswap, fcopy, faxpy, faxpby:
		// vector X
		if ind.IncX <= 0 {
			return errors.New("incX illegal, <=0")
		}
		if ind.OffsetX < 0 {
			return errors.New("offsetX illegal, <0")
		}
		sizeX := X.NumElements()
		if sizeX >= ind.OffsetX + 1 {
			// calculate default size for N based on X size
			nX = 1 + (sizeX - ind.OffsetX - 1)/ind.IncX
		}
		if sizeX < ind.OffsetX + 1 + (ind.Nx-1)*abs(ind.IncX) {
			return errors.New("X size error")
		}
		if ind.Nx < 0 {
			ind.Nx = nX
		}
		// vector Y
		if ind.IncY <= 0 {
			return errors.New("incY illegal, <=0")
		}
		if ind.OffsetY < 0 {
			return errors.New("offsetY illegal, <0")
		}
		sizeY := Y.NumElements()
		if sizeY >= ind.OffsetY + 1 {
			// calculate default size for N based on Y size
			nY = 1 + (sizeY - ind.OffsetY - 1)/ind.IncY
		}
		if ind.Ny < 0 {
			ind.Ny = nY
		}
		if sizeY < ind.OffsetY + 1 + (ind.Ny-1)*abs(ind.IncY) {
			//fmt.Printf("sizeY=%d, inds: %#v\n", sizeY, ind)
			return errors.New("Y size error")
		}

	case frotg, frotmg, frot, frotm:
	}
	return nil
}

func check_level2_func(ind *linalg.IndexOpts, fn funcNum, X, Y, A matrix.Matrix, pars *linalg.Parameters) error {
	if ind.IncX <= 0 {
		return errors.New("incX")
	}
	if ind.IncY <= 0 {
		return errors.New("incY")
	}

	sizeA := A.NumElements()
	switch fn {
	case fgemv:		// general matrix
		if ind.M < 0 {
			ind.M = A.Rows()
		}
		if ind.N < 0 {
			ind.N = A.Cols()
		}
		if ind.LDa == 0 {
			ind.LDa = max(1, A.Rows())
		}
		if ind.OffsetA < 0 {
			return errors.New("offsetA")
		}
		if ind.N > 0 && ind.M > 0 &&
			sizeA < ind.OffsetA + (ind.N-1)*ind.LDa + ind.M {
 			return errors.New("sizeA")
		}
		if ind.OffsetX < 0 {
			return errors.New("offsetX")
		}
		if ind.OffsetY < 0 {
			return errors.New("offsetY")
		}
		sizeX := X.NumElements()
		sizeY := Y.NumElements()
		if pars.Trans == linalg.PNoTrans {
			if ind.N > 0 && sizeX < ind.OffsetX + (ind.N-1)*abs(ind.IncX) + 1 {
 				return errors.New("sizeX")
			}
			if ind.M > 0 && sizeY < ind.OffsetY + (ind.M-1)*abs(ind.IncY) + 1 {
 				return errors.New("sizeY")
			}
		} else {
			if ind.M > 0 && sizeX < ind.OffsetX + (ind.M-1)*abs(ind.IncX) + 1 {
 				return errors.New("sizeX")
			}
			if ind.N > 0 && sizeY < ind.OffsetY + (ind.N-1)*abs(ind.IncY) + 1 {
 				return errors.New("sizeY")
			}
		}
	case fger:
		if ind.M < 0 {
			ind.M = A.Rows()
		}
		if ind.N < 0 {
			ind.N = A.Cols()
		}
		if ind.M == 0 || ind.N == 0 {
			return nil
		}
		if ind.M > 0 && ind.N > 0 {
			if ind.LDa == 0 {
				ind.LDa = max(1, A.Rows())
			}
			if ind.LDa < max(1, ind.M) {
				return errors.New("ldA")
			}
			if ind.OffsetA < 0 {
				return errors.New("offsetA")
			}
			if sizeA < ind.OffsetA + (ind.N-1)*ind.LDa + ind.M {
 				return errors.New("sizeA")
			}
			if ind.OffsetX < 0 {
				return errors.New("offsetX")
			}
			if ind.OffsetY < 0 {
				return errors.New("offsetY")
			}
			sizeX := X.NumElements()
			if sizeX < ind.OffsetX + (ind.M-1)*abs(ind.IncX) + 1 {
 				return errors.New("sizeX")
			}
			sizeY := Y.NumElements()
			if sizeY < ind.OffsetY + (ind.N-1)*abs(ind.IncY) + 1 {
 				return errors.New("sizeY")
			}
		}			
	case fgbmv:		// general banded
		if ind.M < 0 {
			ind.M = A.Rows()
		}
		if ind.N < 0 {
			ind.N = A.Cols()
		}
		if ind.Kl < 0 {
			return errors.New("kl")
		}
		if ind.Ku < 0 {
			ind.Ku = A.Rows() - 1 - ind.Kl
		}
		if ind.Ku < 0 {
			return errors.New("ku")
		}
		if ind.LDa == 0 {
			ind.LDa = max(1, A.Rows())
		}
		if ind.LDa < ind.Kl + ind.Ku + 1 {
			return errors.New("ldA")
		}
		if ind.OffsetA < 0 {
			return errors.New("offsetA")
		}
		sizeA := A.NumElements()
		if ind.N > 0 && ind.M > 0 &&
			sizeA < ind.OffsetA + (ind.N-1)*ind.LDa + ind.Kl + ind.Ku + 1 {
 			return errors.New("sizeA")
		}
		if ind.OffsetX < 0 {
			return errors.New("offsetX")
		}
		if ind.OffsetY < 0 {
			return errors.New("offsetY")
		}
		sizeX := X.NumElements()
		sizeY := Y.NumElements()
		if pars.Trans == linalg.PNoTrans {
			if ind.N > 0 && sizeX < ind.OffsetX + (ind.N-1)*abs(ind.IncX) + 1 {
 				return errors.New("sizeX")
			}
			if ind.N > 0 && sizeY < ind.OffsetY + (ind.M-1)*abs(ind.IncY) + 1 {
 				return errors.New("sizeY")
			}
		} else {
			if ind.N > 0 && sizeX < ind.OffsetX + (ind.M-1)*abs(ind.IncX) + 1 {
 				return errors.New("sizeX")
			}
			if ind.N > 0 && sizeY < ind.OffsetY + (ind.N-1)*abs(ind.IncY) + 1 {
 				return errors.New("sizeY")
			}
		}
	case ftrmv, ftrsv:
		// ftrmv = triangular 
		// ftrsv = triangular solve
		if ind.N < 0 {
			if A.Rows() != A.Cols() {
				return errors.New("A not square")
			}
			ind.N = A.Rows()
		}
		if ind.N > 0 {
			if ind.LDa == 0 {
				ind.LDa = max(1, A.Rows())
			}
			if ind.LDa < max(1, ind.N) {
				return errors.New("ldA")
			}
			if ind.OffsetA < 0 {
				return errors.New("offsetA")
			}
			sizeA := A.NumElements()
			if sizeA < ind.OffsetA + (ind.N-1)*ind.LDa + ind.N  {
 				return errors.New("sizeA")
			}
			sizeX := X.NumElements()
			if sizeX < ind.OffsetX + (ind.N-1)*abs(ind.IncX) + 1 {
 				return errors.New("sizeX")
			}
		}
	case ftbmv, ftbsv, fsbmv:
		// ftbmv = triangular banded
		// ftbsv = triangular banded solve
		// fsbmv = symmetric banded product
		if ind.N < 0 {
			ind.N = A.Rows()
		}
		if ind.N > 0 {
			if ind.K < 0 {
				ind.K = max(0, A.Rows()-1)
			}
			if ind.LDa == 0 {
				ind.LDa = max(1, A.Rows())
			}
			if ind.LDa < ind.K + 1 {
				return errors.New("ldA")
			}
			if ind.OffsetA < 0 {
				return errors.New("offsetA")
			}
			sizeA := A.NumElements()
			if sizeA < ind.OffsetA + (ind.N-1)*ind.LDa + ind.K + 1  {
 				return errors.New("sizeA")
			}
			sizeX := X.NumElements()
			if sizeX < ind.OffsetX + (ind.N-1)*abs(ind.IncX) + 1 {
 				return errors.New("sizeX")
			}
			if Y != nil {
				sizeY := Y.NumElements()
				if sizeY < ind.OffsetY + (ind.N-1)*abs(ind.IncY) + 1 {
 					return errors.New("sizeY")
				}
			}
		}
	case fsymv, fsyr, fsyr2:
		// fsymv = symmetric product
		// fsyr = symmetric rank update
		// fsyr2 = symmetric rank-2 update
		if ind.N < 0 {
			if A.Rows() != A.Cols() {
				return errors.New("A not square")
			}
			ind.N = A.Rows()
		}
		if ind.N > 0 {
			if ind.LDa == 0 {
				ind.LDa = max(1, A.Rows())
			}
			if ind.LDa < max(1, ind.N) {
				return errors.New("ldA")
			}
			if ind.OffsetA < 0 {
				return errors.New("offsetA")
			}
			sizeA := A.NumElements()
			if sizeA < ind.OffsetA + (ind.N-1)*ind.LDa + ind.N {
 				return errors.New("sizeA")
			}
			if ind.OffsetX < 0 {
				return errors.New("offsetX")
			}
			sizeX := X.NumElements()
			if sizeX < ind.OffsetX + (ind.N-1)*abs(ind.IncX) + 1 {
 				return errors.New("sizeX")
			}
			if Y != nil {	
				if ind.OffsetY < 0 {
					return errors.New("offsetY")
				}
				sizeY := Y.NumElements()
				if sizeY < ind.OffsetY + (ind.N-1)*abs(ind.IncY) + 1 {
 					return errors.New("sizeY")
				}
			}
		}				
	case fspr, fdspr2, ftpsv, fspmv, ftpmv:
		// ftpsv = triangular packed solve
		// fspmv = symmetric packed product
		// ftpmv = triangular packed
	}
	return nil
}

func check_level3_func(ind *linalg.IndexOpts, fn funcNum, A, B, C matrix.Matrix,
	pars *linalg.Parameters) (err error) {	

	switch fn {
	case fgemm:
		if ind.M < 0 {
			if pars.TransA == linalg.PNoTrans {
				ind.M = A.Rows()
			} else {
				ind.M = A.Cols()
			}
		}
		if ind.N < 0 {
			if pars.TransB == linalg.PNoTrans {
				ind.N = B.Cols()
			} else {
				ind.N = B.Rows()
			}
		}
		if ind.M == 0 || ind.N == 0 {
			return nil
		}
		if ind.K < 0 {
			if pars.TransA == linalg.PNoTrans {
				ind.K = A.Cols()
			} else {
				ind.K = A.Rows()
			}
			if pars.TransB == linalg.PNoTrans && ind.K != B.Rows() ||
				pars.TransB != linalg.PNoTrans && ind.K != B.Cols() {
				return errors.New("dimensions of A and B do not match")
			}
		}
		if ind.OffsetA < 0 { 
			return errors.New("offsetA illegal, <0")
		}
		if ind.LDa == 0 {
			ind.LDa = max(1, A.Rows())
		}
		if ind.K > 0 {
			if (pars.TransA == linalg.PNoTrans && ind.LDa < max(1, ind.M)) ||
				(pars.TransA != linalg.PNoTrans && ind.LDa < max(1, ind.K)) {
				return errors.New("inconsistent ldA")
			}
			sizeA := A.NumElements()
			if (pars.TransA == linalg.PNoTrans &&
				sizeA < ind.OffsetA + (ind.K-1)*ind.LDa + ind.M) ||
				(pars.TransA != linalg.PNoTrans &&
				sizeA < ind.OffsetA + (ind.M-1)*ind.LDa + ind.K) {
				return errors.New("sizeA")
			}
		}
		// B matrix
		if ind.OffsetB < 0 { 
			return errors.New("offsetB illegal, <0")
		}
		if ind.LDb == 0 {
			ind.LDb = max(1, B.Rows())
		}
		if ind.K > 0 {
			if (pars.TransB == linalg.PNoTrans && ind.LDb < max(1, ind.K)) ||
				(pars.TransB != linalg.PNoTrans && ind.LDb < max(1, ind.N)) {
				return errors.New("inconsistent ldB")
			}
			sizeB := B.NumElements()
			if (pars.TransB == linalg.PNoTrans &&
				sizeB < ind.OffsetB + (ind.N-1)*ind.LDb + ind.K) ||
				(pars.TransB != linalg.PNoTrans &&
				sizeB < ind.OffsetB + (ind.K-1)*ind.LDb + ind.N) {
				return errors.New("sizeB")
			}
		}
		// C matrix
		if ind.OffsetC < 0 { 
			return errors.New("offsetC illegal, <0")
		}
		if ind.LDc == 0 {
			ind.LDc = max(1, C.Rows())
		}
		if ind.LDc < max(1, ind.M) {
			return errors.New("inconsistent ldC")
		}
		sizeC := C.NumElements()
		if sizeC < ind.OffsetC + (ind.N-1)*ind.LDc + ind.M {
			return errors.New("sizeC")
		}

	case fsymm, ftrmm, ftrsm: 
		if ind.M < 0 {
			ind.M = B.Rows()
			if pars.Side == linalg.PLeft && (ind.M != A.Rows() || ind.M != A.Cols()) {
				return errors.New("dimensions of A and B do not match")
			}
		}
		if ind.N < 0 {
			ind.N = B.Cols()
			if pars.Side == linalg.PRight && (ind.N != A.Rows() || ind.N != A.Cols()) {
				return errors.New("dimensions of A and B do not match")
			}
		}
		if ind.M == 0 || ind.N == 0 {
			return
		}
		// check A
		if ind.OffsetB < 0 { 
			return errors.New("offsetB illegal, <0")
		}
		if ind.LDa == 0 {
			ind.LDa = max(1, A.Rows())
		}
		if pars.Side == linalg.PLeft && ind.LDa < max(1, ind.M) || ind.LDa < max(1, ind.N) {
			return errors.New("ldA")
		}
		sizeA := A.NumElements()
		if (pars.Side == linalg.PLeft && sizeA < ind.OffsetA + (ind.M-1)*ind.LDa + ind.M) ||
			(pars.Side == linalg.PRight && sizeA < ind.OffsetA + (ind.N-1)*ind.LDa + ind.N) {
			return errors.New("sizeA")
		}

		if B != nil {
			if ind.OffsetB < 0 { 
				return errors.New("offsetB illegal, <0")
			}
			if ind.LDb == 0 {
				ind.LDb = max(1, B.Rows())
			}
			if ind.LDb < max(1, ind.M) {
				return errors.New("ldB")
			}
			sizeB := B.NumElements()
			if sizeB < ind.OffsetB + (ind.N-1)*ind.LDb + ind.M {
				return errors.New("sizeB")
			}
		}
			
		if C != nil {
			if ind.OffsetC < 0 { 
				return errors.New("offsetC illegal, <0")
			}
			if ind.LDc == 0 {
				ind.LDc = max(1, C.Rows())
			}
			if ind.LDc < max(1, ind.M) {
				return errors.New("ldC")
			}
			sizeC := C.NumElements()
			if sizeC < ind.OffsetC + (ind.N-1)*ind.LDc + ind.M {
				return errors.New("sizeC")
			}
		}
	case fsyrk: 
		if ind.N < 0 {
			if pars.Trans == linalg.PNoTrans {
				ind.N = A.Rows()
			} else {
				ind.N = A.Cols()
			}
		}
		if ind.K < 0 {
			if pars.Trans == linalg.PNoTrans {
				ind.K = A.Cols()
			} else {
				ind.K = A.Rows()
			}
		}
		if ind.N == 0 {
			return
		}
		if ind.LDa == 0 {
			ind.LDa = max(1, A.Rows())
		}
		if ind.OffsetA < 0 { 
			return errors.New("offsetA")
		}
		if ind.K > 0 {
			if (pars.Trans == linalg.PNoTrans && ind.LDa < max(1, ind.N)) ||
				(pars.Trans != linalg.PNoTrans && ind.LDa < max(1, ind.K)) {
				return errors.New("inconsistent ldA")
			}
			sizeA := A.NumElements()
			if (pars.Trans == linalg.PNoTrans &&
				sizeA < ind.OffsetA + (ind.K-1)*ind.LDa + ind.N) ||
				(pars.TransA != linalg.PNoTrans &&
				sizeA < ind.OffsetA + (ind.N-1)*ind.LDa + ind.K) {
				return errors.New("sizeA")
			}
		}
			
		if ind.OffsetC < 0 { 
			return errors.New("offsetC illegal, <0")
		}
		if ind.LDc == 0 {
			ind.LDc = max(1, C.Rows())
		}
		if ind.LDc < max(1, ind.N) {
			return errors.New("ldC")
		}
		sizeC := C.NumElements()
		if sizeC < ind.OffsetC + (ind.N-1)*ind.LDc + ind.N {
			return errors.New("sizeC")
		}
	case fsyr2k: 
		if ind.N < 0 {
			if pars.Trans == linalg.PNoTrans {
				ind.N = A.Rows()
				if ind.N != B.Rows() {
					return errors.New("dimensions of A and B do not match")
				}
			} else {
				ind.N = A.Cols()
				if ind.N != B.Cols() {
					return errors.New("dimensions of A and B do not match")
				}
			}
		}
		if ind.N == 0 {
			return
		}
		if ind.K < 0 {
			if pars.Trans == linalg.PNoTrans {
				ind.K = A.Cols()
				if ind.K != B.Cols() {
					return errors.New("dimensions of A and B do not match")
				}
			} else {
				ind.K = A.Rows()
				if ind.K != B.Rows() {
					return errors.New("dimensions of A and B do not match")
				}
			}
		}
		if ind.LDa == 0 {
			ind.LDa = max(1, A.Rows())
		}
		if ind.K > 0 {
			if (pars.Trans == linalg.PNoTrans && ind.LDa < max(1, ind.N)) ||
				(pars.Trans != linalg.PNoTrans && ind.LDa < max(1, ind.K)) {
				return errors.New("inconsistent ldA")
			}
			sizeA := A.NumElements()
			if (pars.Trans == linalg.PNoTrans &&
				sizeA < ind.OffsetA + (ind.K-1)*ind.LDa + ind.N) ||
				(pars.TransA != linalg.PNoTrans &&
				sizeA < ind.OffsetA + (ind.N-1)*ind.LDa + ind.K) {
				return errors.New("sizeA")
			}
		}
		if ind.OffsetB < 0 { 
			return errors.New("offsetB illegal, <0")
		}
		if ind.LDb == 0 {
			ind.LDb = max(1, B.Rows())
		}
		if ind.K > 0 {
			if (pars.Trans == linalg.PNoTrans && ind.LDb < max(1, ind.N)) ||
				(pars.Trans != linalg.PNoTrans && ind.LDb < max(1, ind.K)) {
				return errors.New("ldB")
			}
			sizeB := B.NumElements()
			if (pars.Trans == linalg.PNoTrans &&
				sizeB < ind.OffsetB + (ind.K-1)*ind.LDb + ind.N) ||
				(pars.Trans != linalg.PNoTrans &&
				sizeB < ind.OffsetB + (ind.N-1)*ind.LDb + ind.K) {
				return errors.New("sizeB")
			}
		}
		if ind.OffsetC < 0 { 
			return errors.New("offsetC illegal, <0")
		}
		if ind.LDc == 0 {
			ind.LDc = max(1, C.Rows())
		}
		if ind.LDc < max(1, ind.N) {
			return errors.New("ldC")
		}
		sizeC := C.NumElements()
		if sizeC < ind.OffsetC + (ind.N-1)*ind.LDc + ind.N {
			return errors.New("sizeC")
		}
	}
	err = nil
	return
}

// Local Variables:
// tab-width: 4
// End:
