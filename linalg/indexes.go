
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package linalg

import (
	"strings"
	"fmt"
)

// Type Opt holds one BLAS/LAPACK index or parameter option.
type Opt struct {
	Name string
	Val  int
}

// LinalgIndex structure holds fields for various BLAS/LAPACK indexing
// variables.
type LinalgIndex struct {
	// these for BLAS and LAPACK
	N, Nx, Ny int
	M, Ma, Mb int
	LDa, LDb, LDc int
	IncX, IncY int
	OffsetX, OffsetY, OffsetA, OffsetB, OffsetC int
	K int
	Ku int
	Kl int
	// these used in LAPACK
	Nrhs int
	OffsetD, OffsetDL, OffsetDU int
	LDw, LDz int
	OffsetW, OffsetZ int
}

// Parse option list and return index structure with relevant fields set and
// other fields with default values.
func GetIndexOpts(opts ...Opt) *LinalgIndex {
	is := &LinalgIndex{
		-1, -1, -1,				// n, nX, nY
		-1, -1, -1,				// m, mA, mB
		 0,  0,  0,				// ldA, ldB, ldC
		 1,  1,					// incX, incY
		 0,  0,  0,  0,	 0,		// offsetX, ... offsetC
		-1, -1,  0,				// k, ku, kl
		-1,						// nrhs
		 0,  0,  0,				// offsetD, offsetDL, OffsetDU,
		 0,  0,					// LDw, LDz
		 0,  0,					// OffsetW, OffsetZ
	}

	for _, o := range opts {
		switch {
		case strings.EqualFold(o.Name, "inc"):
			is.IncX = o.Val; is.IncY = o.Val
		case strings.EqualFold(o.Name, "incx"):
			is.IncX = o.Val
		case strings.EqualFold(o.Name, "incy"):
			is.IncY = o.Val
		case strings.EqualFold(o.Name, "lda"):
			is.LDa = o.Val
		case strings.EqualFold(o.Name, "ldb"):
			is.LDb = o.Val
		case strings.EqualFold(o.Name, "ldw"):
			is.LDw = o.Val
		case strings.EqualFold(o.Name, "ldz"):
			is.LDz = o.Val
		case strings.EqualFold(o.Name, "offset"):
			is.OffsetX = o.Val; is.OffsetY = o.Val
			is.OffsetA = o.Val; is.OffsetB = o.Val
			is.OffsetC = o.Val
		case strings.EqualFold(o.Name, "offsetx"):
			is.OffsetX = o.Val
		case strings.EqualFold(o.Name, "offsety"):
			is.OffsetY = o.Val
		case strings.EqualFold(o.Name, "offseta"):
			is.OffsetA = o.Val
		case strings.EqualFold(o.Name, "offsetb"):
			is.OffsetB = o.Val
		case strings.EqualFold(o.Name, "offsetc"):
			is.OffsetC = o.Val
		case strings.EqualFold(o.Name, "offsetw"):
			is.OffsetW = o.Val
		case strings.EqualFold(o.Name, "offsetd"):
			is.OffsetD = o.Val
		case strings.EqualFold(o.Name, "offsetdl"):
			is.OffsetDL = o.Val
		case strings.EqualFold(o.Name, "offsetdu"):
			is.OffsetDU = o.Val
		case strings.EqualFold(o.Name, "offsetdw"):
			is.OffsetW = o.Val
		case strings.EqualFold(o.Name, "offsetdz"):
			is.OffsetZ = o.Val
		case strings.EqualFold(o.Name, "n"):
			is.N = o.Val; is.Nx = o.Val; is.Ny = o.Val
		case strings.EqualFold(o.Name, "nx"):
			is.Nx = o.Val
		case strings.EqualFold(o.Name, "ny"):
			is.Ny = o.Val
		case strings.EqualFold(o.Name, "m"):
			is.M = o.Val; is.Ma = o.Val; is.Mb = o.Val
		case strings.EqualFold(o.Name, "ma"):
			is.Ma = o.Val
		case strings.EqualFold(o.Name, "mb"):
			is.Mb = o.Val
		case strings.EqualFold(o.Name, "k"):
			is.K = o.Val
		case strings.EqualFold(o.Name, "kl"):
			is.Kl = o.Val
		case strings.EqualFold(o.Name, "ku"):
			is.Ku = o.Val
		case strings.EqualFold(o.Name, "nrhs"):
			is.Nrhs = o.Val
		}
	}
	return is
}

func PrintIndexes(p *LinalgIndex) {
	// these used in BLAS/LAPACK
	fmt.Printf("N=%d, Nx=%d, Ny=%d\n", p.N, p.Nx, p.Ny)
	fmt.Printf("M=%d, Ma=%d, Mb=%d\n", p.M, p.Ma, p.Mb)
	fmt.Printf("LDa=%d, LDb=%d, LDc=%d\n", p.LDa, p.LDb, p.LDc)
	fmt.Printf("IncX=%d, IncY=%d\n", p.IncX, p.IncY)
	fmt.Printf("Ox=%d, Oy=%d, Oa=%d, Ob=%d, Oc=%d\n",
		p.OffsetX, p.OffsetY, p.OffsetA, p.OffsetB, p.OffsetC)
	fmt.Printf("K=%d, Ku=%d, Kl=%d\n", p.K,	p.Ku, p.Kl)
	// these used in LAPACK
	fmt.Printf("NRHS=%d\n", p.Nrhs)
	fmt.Printf("Od=%d, Odl=%d, Odu=%d\n", p.OffsetD, p.OffsetDL, p.OffsetDU)
	fmt.Printf("LDw=%d, LDz=%d\n", p.LDw, p.LDz)
	fmt.Printf("Ow=%d, Oz=%d\n", p.OffsetW, p.OffsetZ)

}

// Local Variables:
// tab-width: 4
// End:
