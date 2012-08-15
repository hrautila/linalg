
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/cvx package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.


package main

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/cvx"
	"fmt"
)

func main() {

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

	g0 := matrix.FloatMatrixStacked(gdata0, matrix.ColumnOrder)
	g1 := matrix.FloatMatrixStacked(gdata1, matrix.ColumnOrder)
	Ghs := cvx.FloatSetNew("Gs", "hs")
	Ghs.Append("Gs", g0, g1)

	h0 := matrix.FloatMatrixStacked(hdata0, matrix.ColumnOrder)
	h1 := matrix.FloatMatrixStacked(hdata1, matrix.ColumnOrder)
	Ghs.Append("hs", h0, h1)

	c := matrix.FloatVector([]float64{1.0, -1.0, 1.0})
	Ghs.Print()
	fmt.Printf("calling...\n")
	// nil variables
	var Gs, hs, A, b *matrix.FloatMatrix = nil, nil, nil, nil

	var solopts cvx.SolverOptions
	solopts.MaxIter = 30
	solopts.ShowProgress = true
	sol, err := cvx.Sdp(c, Gs, hs, A, b, Ghs, &solopts, nil, nil)
	if sol != nil && sol.Status == cvx.Optimal {
		x := sol.Result.At("x")[0]
		fmt.Printf("x=\n%v\n", x.ToString("%.9f"))
		for i, m := range sol.Result.At("ss") {
			fmt.Printf("ss[%d]=\n%v\n", i, m.ToString("%.9f"))
		}
		for i, m := range sol.Result.At("zs") {
			fmt.Printf("zs[%d]=\n%v\n", i, m.ToString("%.9f"))
		}
	} else {
		fmt.Printf("status: %v\n", err)
	}
}

// Local Variables:
// tab-width: 4
// End:
