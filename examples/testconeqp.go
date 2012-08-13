
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.


package main

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/cvx"
	"fmt"
)

func main() {
	arows := [][]float64{
		[]float64{ 0.3, -0.4, -0.2, -0.4,  1.3},
		[]float64{ 0.6,  1.2, -1.7,  0.3, -0.3},
		[]float64{-0.3,  0.0,  0.6, -1.2, -2.0}}
		

	A := matrix.FloatMatrixStacked(arows, false)
	b := matrix.FloatVector([]float64{1.5, 0.0, -1.2, -0.7, 0.0})

	_, n := A.Size()
	N := n + 1 + n

	h := matrix.FloatZeros(N, 1)
	h.SetIndex(n, 1.0)

	I0 := matrix.FloatDiagonal(n, -1.0)
	I1 := matrix.FloatIdentity(n)
	G := matrix.FloatCombined(I0, matrix.FloatZeros(1, n), I1)

	At := A.Transpose()
	P := At.Times(A)
	q := At.Times(b).Neg()

	dims := cvx.DSetNew("l", "q", "s")
	dims.Set("l", []int{n})
	dims.Set("q", []int{n+1})

	var solopts cvx.SolverOptions
	solopts.MaxIter = 10
	solopts.ShowProgress = true
	sol, err := cvx.ConeQp(P, q, G, h, nil, nil, dims, &solopts, nil)
	if err == nil {
		fmt.Printf("Optimal\n")
		fmt.Printf("x=\n%v\n", sol.X.ConvertToString())
		fmt.Printf("s=\n%v\n", sol.S.ConvertToString())
		fmt.Printf("z=\n%v\n", sol.Z.ConvertToString())
	}
}



// Local Variables:
// tab-width: 4
// End:
