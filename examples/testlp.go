
package main

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/cvx"
	"fmt"
)

func main() {

	gdata := [][]float64{
		[]float64{ 2.0, 1.0, -1.0,  0.0 },
		[]float64{ 1.0, 2.0,  0.0, -1.0 }}

	c := matrix.FloatVector([]float64{-4.0, -5.0})
	G := matrix.FloatMatrixStacked(gdata, matrix.ColumnOrder)
	h := matrix.FloatVector([]float64{3.0, 3.0, 0.0, 0.0})

	fmt.Printf("G=\n%v\n", G.ToString("%.2f"))
	fmt.Printf("h=\n%v\n", h.ToString("%.2f"))

	var solopts cvx.SolverOptions
	solopts.MaxIter = 30
	solopts.ShowProgress = true
	sol, err := cvx.Lp(c, G, h, nil, nil, &solopts, nil, nil)
	fmt.Printf("status: %v\n", err)
	if sol != nil && sol.Status == cvx.Optimal {
		fmt.Printf("x=\n%v\n", sol.Result.At("x")[0].ToString("%.9f"))
		fmt.Printf("s=\n%v\n", sol.Result.At("s")[0].ToString("%.9f"))
		fmt.Printf("z=\n%v\n", sol.Result.At("z")[0].ToString("%.9f"))
	}
}

// Local Variables:
// tab-width: 4
// End:
