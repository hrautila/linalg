

package blas

import (
	"github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/matrix"
	"errors"
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


// index structure holds fields for various BLAS indexing variables.
type index struct {
	// these for BLAS and LAPACK
	N, Nx, Ny int
	M, Ma, Mb int
	LDa, LDb, LDc int
	IncX, IncY int
	OffsetX, OffsetY, OffsetA, OffsetB, OffsetC int
	K int
	Ku int
	Kl int
}

var indexNames map[string]int = {
	"inc": 1, "incx": 1, "incy": 1,
	"lda":0, "ldb":0, "ldc":0,
	"offsetx":0, "offsety":0, "offseta":0, "offsetb":0, "offsetc":0,
	"n":-1, "nx":-1, "ny":-1,
	"m":-1, "k":-1, "ku":-1, "kl":0}

func getIndexOpts(opts ...linalg.Option) *index {
	is := &index{
		-1, -1, -1,				// n, nX, nY
		-1, -1, -1,				// m, mA, mB
		 0,  0,  0,				// ldA, ldB, ldC
		 1,  1,					// incX, incY
		 0,  0,  0,  0,	 0,		// offsetX, ... offsetC
		-1, -1,  0,				// k, ku, kl
		-1,						// nrhs
	}
	for _, o := range opts {
		name := strings.ToLower(o.Name())
		if val, e := indexNames[name]; e == nil {
			oval := o.Int()
			switch name {
			case "inc":
				is.Inc = o.Int()
				is.Incx = is.Inc; is.Incy = is.Inc
			case "incy":
				is.IncY = o.Int()
			case "incx":
				is.IncX = o.Int()
			case "lda":
				is.LDa = o.Int()
			case "ldb":
				is.LDb = o.Int()
			case "ldc":
				is.LDc = o.Int()
			case "n":
				is.N = o.Int()
				is.Nx = is.N; is.Ny = is.N
			case "m":
				is.M = o.Int()
				is.Ma = is.M; is.Mb = is.M
			case "offset", "offsetx":
				is.OffsetX = o.Int()
			case "offsety":
				is.OffsetY = o.Int()
			case "offseta":
				is.OffsetA = o.Int()
			case "offsetb":
				is.OffsetB = o.Int()
			case "offsetc":
				is.OffsetC = o.Int()
			case "k":
				is.K = o.Int()
			case "kl":
				is.Kl = o.Int()
			case "ku":
				is.Ku = o.Int()
			}
		}
	}
}
	


func check_level1_func(ind *linalg.LinalgIndex, fn funcNum, X, Y matrix.Matrix) error {

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
		if sizeY < ind.OffsetY + 1 + (ind.Ny-1)*abs(ind.IncY) {
			return errors.New("Y size error")
		}
		if ind.Ny < 0 {
			ind.Ny = nY
		}

	case frotg, frotmg, frot, frotm:
	}
	return nil
}

func check_level2_func(ind *linalg.LinalgIndex, fn funcNum, X, Y, A matrix.Matrix, pars *linalg.Parameters) error {
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
			if sizeY < ind.OffsetY + (ind.M-1)*abs(ind.IncY) + 1 {
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

func check_level3_func(ind *linalg.LinalgIndex, fn funcNum, A, B, C matrix.Matrix,
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
				ind.N = A.Cols()
			} else {
				ind.N = A.Rows()
			}
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
		if ind.OffsetB < 0 { 
			return errors.New("offsetB illegal, <0")
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
		if B != nil {
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
		}
		// C matrix
		if C != nil {
			if ind.OffsetC < 0 { 
				return errors.New("offsetC illegal, <0")
			}
			if ind.LDc == 0 {
				ind.LDb = max(1, C.Rows())
			}
			if ind.LDc < max(1, ind.M) {
				return errors.New("inconsistent ldC")
			}
			sizeC := C.NumElements()
			if sizeC < ind.OffsetC + (ind.N-1)*ind.LDc + ind.M {
				return errors.New("sizeC")
			}
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
	case fsyrk, fsyr2k: 
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
		if B != nil {
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
		}
			
		if C != nil {
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
			if sizeC < ind.OffsetC + (ind.N-1)*ind.LDc + ind.M {
				return errors.New("sizeC")
			}
		}
	}
	err = nil
	return
}

// Local Variables:
// tab-width: 4
// End:
