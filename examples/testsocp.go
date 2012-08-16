
package main

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/cvx"
	"fmt"
)

func main() {

	gdata0 := [][]float64{
		[]float64{12., 13.,  12.},
		[]float64{ 6., -3., -12.},
		[]float64{-5., -5.,   6.}}

    gdata1 := [][]float64{
		[]float64{ 3.,  3., -1.,  1.},
        []float64{-6., -6., -9., 19.},
        []float64{10., -2., -2., -3.}} 


	c := matrix.FloatVector([]float64{-2.0, 1.0, 5.0})
	g0 := matrix.FloatMatrixStacked(gdata0, matrix.ColumnOrder)
	g1 := matrix.FloatMatrixStacked(gdata1, matrix.ColumnOrder)
	Ghq := cvx.FloatSetNew("Gq", "hq")
	Ghq.Append("Gq", g0, g1)

	h0 := matrix.FloatVector([]float64{-12.0, -3.0, -2.0})
	h1 := matrix.FloatVector([]float64{ 27.0,  0.0,  3.0, -42.0})
	Ghq.Append("hq", h0, h1)

	Ghq.Print()

	var Gl, hl, A, b *matrix.FloatMatrix = nil, nil, nil, nil
	var solopts cvx.SolverOptions
	solopts.MaxIter = 30
	solopts.ShowProgress = true
	sol, err := cvx.Socp(c, Gl, hl, A, b, Ghq, &solopts, nil, nil)
	fmt.Printf("status: %v\n", err)
	if sol != nil && sol.Status == cvx.Optimal {
		x := sol.Result.At("x")[0]
		fmt.Printf("x=\n%v\n", x.ToString("%.9f"))
		for i, m := range sol.Result.At("sq") {
			fmt.Printf("sq[%d]=\n%v\n", i, m.ToString("%.9f"))
		}
		for i, m := range sol.Result.At("zq") {
			fmt.Printf("zq[%d]=\n%v\n", i, m.ToString("%.9f"))
		}
	}
}

// Local Variables:
// tab-width: 4
// End:
