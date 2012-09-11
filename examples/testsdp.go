
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/cvx package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.


package main

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/cvx"
	"github.com/hrautila/go.opt/cvx/sets"
	"github.com/hrautila/go.opt/cvx/checkpnt"
	"fmt"
	"flag"
)

var xVal, ss0Val, ss1Val, zs0Val, zs1Val string
var spPath string
var maxIter int 
var spVerbose bool

func init() {
	flag.BoolVar(&spVerbose, "V", false, "Savepoint verbose reporting.")
	flag.IntVar(&maxIter, "N", -1, "Max number of iterations.")
	flag.StringVar(&spPath, "sp", "", "savepoint directory")
	flag.StringVar(&xVal, "x", "", "Reference value for X")
	flag.StringVar(&ss0Val, "ss0", "", "Reference value for SQ[0]")
	flag.StringVar(&ss1Val, "ss1", "", "Reference value for SQ[1]")
	flag.StringVar(&zs0Val, "zs0", "", "Reference value for ZQ[0]")
	flag.StringVar(&zs1Val, "zs1", "", "Reference value for ZQ[1]")
}
	
func error(ref, val *matrix.FloatMatrix) (nrm float64, diff *matrix.FloatMatrix) {
	diff = ref.Minus(val)
	nrm = blas.Nrm2(diff).Float()
	return
}

func check(x, ss0, ss1, zs0, zs1 *matrix.FloatMatrix) {
	if len(xVal) > 0 {
		ref, _ := matrix.FloatParseSpe(xVal)
		nrm, diff := error(ref, x)
		fmt.Printf("x: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
	if len(ss0Val) > 0 {
		ref, _ := matrix.FloatParseSpe(ss0Val)
		nrm, diff := error(ref,ss0)
		fmt.Printf("ss0: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
	if len(ss1Val) > 0 {
		ref, _ := matrix.FloatParseSpe(ss1Val)
		nrm, diff := error(ref,ss1)
		fmt.Printf("ss1: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
	if len(zs0Val) > 0 {
		ref, _ := matrix.FloatParseSpe(zs0Val)
		nrm, diff := error(ref, zs0)
		fmt.Printf("zs0: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
	if len(zs1Val) > 0 {
		ref, _ := matrix.FloatParseSpe(zs1Val)
		nrm, diff := error(ref, zs1)
		fmt.Printf("zs1: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
}

func main() {
	flag.Parse()
	if len(spPath) > 0 {
		checkpnt.Reset(spPath)
		checkpnt.Activate()
		checkpnt.Verbose(spVerbose)
		checkpnt.Format("%.17f")
	}

	gdata0 := [][]float64{
		[]float64{-7., -11., -11., 3.},
		[]float64{ 7., -18., -18., 8.},
        []float64{-2.,  -8.,  -8., 1.}}

	gdata1 := [][]float64{
		[]float64{-21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.},  
        []float64{  0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.},  
        []float64{ -5.,   2., -17.,   2.,  -6.,   8., -17.,  -7., 6.}}

	hdata0 := [][]float64{
		[]float64{ 33., -9.},
		[]float64{ -9., 26.}}

	hdata1 := [][]float64{
		[]float64{ 14.,  9., 40.},
		[]float64{  9., 91., 10.},
		[]float64{ 40., 10., 15.}}

	g0 := matrix.FloatMatrixFromTable(gdata0, matrix.ColumnOrder)
	g1 := matrix.FloatMatrixFromTable(gdata1, matrix.ColumnOrder)
	Ghs := sets.FloatSetNew("Gs", "hs")
	Ghs.Append("Gs", g0, g1)

	h0 := matrix.FloatMatrixFromTable(hdata0, matrix.ColumnOrder)
	h1 := matrix.FloatMatrixFromTable(hdata1, matrix.ColumnOrder)
	Ghs.Append("hs", h0, h1)

	c := matrix.FloatVector([]float64{1.0, -1.0, 1.0})

	var Gs, hs, A, b *matrix.FloatMatrix = nil, nil, nil, nil
	var solopts cvx.SolverOptions
	solopts.MaxIter = 30
	if maxIter > 0 {
		solopts.MaxIter = maxIter
	}
	solopts.ShowProgress = true
	sol, err := cvx.Sdp(c, Gs, hs, A, b, Ghs, &solopts, nil, nil)
	if sol != nil && sol.Status == cvx.Optimal {
		x := sol.Result.At("x")[0]
		fmt.Printf("x=\n%v\n", x.ToString("%.9f"))
		/*
		for i, m := range sol.Result.At("ss") {
			fmt.Printf("ss[%d]=\n%v\n", i, m.ToString("%.9f"))
		}
		 */
		for i, m := range sol.Result.At("zs") {
			fmt.Printf("zs[%d]=\n%v\n", i, m.ToString("%.9f"))
		}
		ss0 := sol.Result.At("ss")[0]
		ss1 := sol.Result.At("ss")[1]
		zs0 := sol.Result.At("zs")[0]
		zs1 := sol.Result.At("zs")[1]
		check(x, ss0, ss1, zs0, zs1)
	} else {
		fmt.Printf("status: %v\n", err)
	}
	checkpnt.Report()
}

// Local Variables:
// tab-width: 4
// End:
