
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.


package main

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/cvx"
	"fmt"
	"flag"
)

var xVal, sVal, zVal string

func init() {
	flag.StringVar(&xVal, "x", "", "Reference value for X")
	flag.StringVar(&sVal, "s", "", "Reference value for S")
	flag.StringVar(&zVal, "z", "", "Reference value for Z")
}
	
func error(ref, val *matrix.FloatMatrix) (nrm float64, diff *matrix.FloatMatrix) {
	diff = ref.Minus(val)
	nrm = blas.Nrm2(diff).Float()
	return
}

func check(x, s, z *matrix.FloatMatrix) {
	var xref, sref, zref *matrix.FloatMatrix = nil, nil, nil

	if len(xVal) > 0 {
		xref, _ = matrix.FloatParseSpe(xVal)
		nrm, diff := error(xref, x)
		fmt.Printf("x: nrm=%.17f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.17f"))
		}
	}

	if len(sVal) > 0 {
		sref, _ = matrix.FloatParseSpe(sVal)
		nrm, diff := error(sref, s)
		fmt.Printf("s: nrm=%.17f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.17f"))
		}
	}

	if len(zVal) > 0 {
		zref, _ = matrix.FloatParseSpe(zVal)
		nrm, diff := error(zref, z)
		fmt.Printf("z: nrm=%.17f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.17f"))
		}
	}
}


func main() {
	flag.Parse()
	reftest := flag.NFlag() > 0

	adata := [][]float64{
		[]float64{ 0.3, -0.4, -0.2, -0.4,  1.3},
		[]float64{ 0.6,  1.2, -1.7,  0.3, -0.3},
		[]float64{-0.3,  0.0,  0.6, -1.2, -2.0}}
		
	A := matrix.FloatMatrixStacked(adata, matrix.ColumnOrder)
	b := matrix.FloatVector([]float64{1.5, 0.0, -1.2, -0.7, 0.0})

	_, n := A.Size()
	N := n + 1 + n

	h := matrix.FloatZeros(N, 1)
	h.SetIndex(n, 1.0)

	I0 := matrix.FloatDiagonal(n, -1.0)
	I1 := matrix.FloatIdentity(n)
	G, _ := matrix.FloatMatrixCombined(matrix.StackDown, I0, matrix.FloatZeros(1, n), I1)

	At := A.Transpose()
	P := At.Times(A)
	q := At.Times(b).Scale(-1.0)

	dims := cvx.DSetNew("l", "q", "s")
	dims.Set("l", []int{n})
	dims.Set("q", []int{n+1})

	fmt.Printf("P=\n%v\n", P.ToString("%.15f"))
	fmt.Printf("q=\n%v\n", q.ToString("%.15f"))

	var solopts cvx.SolverOptions
	solopts.MaxIter = 10
	solopts.ShowProgress = true
	sol, err := cvx.ConeQp(P, q, G, h, nil, nil, dims, &solopts, nil)
	if err == nil {
		x := sol.Result.At("x")[0]
		s := sol.Result.At("s")[0]
		z := sol.Result.At("z")[0]
		fmt.Printf("Optimal\n")
		fmt.Printf("x=\n%v\n", x.ToString("%.9f"))
		fmt.Printf("s=\n%v\n", s.ToString("%.9f"))
		fmt.Printf("z=\n%v\n", z.ToString("%.9f"))
		if reftest {
			check(x, s, z)
		}
	}

	/*
	// Reference data from python program. A.T*A and -b.T*A printed with 17decimals
	pdata := [][]float64{
		[]float64{ 2.14000000000000012, -0.47000000000000003, -2.33000000000000007},
		[]float64{-0.47000000000000003,  4.87000000000000011, -0.95999999999999996},
		[]float64{-2.33000000000000007, -0.95999999999999996,  5.88999999999999968}}

	qdata := []float64{	-0.970000000000000, -2.730000000000000, 0.330000000000000}

	Pt := matrix.FloatMatrixStacked(pdata, matrix.ColumnOrder)
	qt := matrix.FloatVector(qdata)
		
	//fmt.Printf("P=\n%v\n", Pt.ToString("%.15f"))
	//fmt.Printf("q=\n%v\n", qt.ToString("%.15f"))
	sol, err = cvx.ConeQp(Pt, qt, G, h, nil, nil, dims, &solopts, nil)
	if err == nil {
		x := sol.Result.At("x")[0]
		s := sol.Result.At("s")[0]
		z := sol.Result.At("z")[0]
		fmt.Printf("Optimal\n")
		fmt.Printf("x=\n%v\n", x.ToString("%.9f"))
		fmt.Printf("s=\n%v\n", s.ToString("%.9f"))
		fmt.Printf("z=\n%v\n", z.ToString("%.9f"))
		if reftest {
			check(x, s, z)
		}
	} else {
		fmt.Printf("status: %s\n", err)
	}
	 */
}



// Local Variables:
// tab-width: 4
// End:
