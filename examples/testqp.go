
package main

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/cvx"
	"github.com/hrautila/go.opt/linalg/blas"
	"fmt"
	"math"
)

func main() {

	Sdata := [][]float64{
		[]float64{ 4e-2,  6e-3, -4e-3,   0.0 },
        []float64{ 6e-3,  1e-2,  0.0,    0.0 },
        []float64{-4e-3,  0.0,   2.5e-3, 0.0 },
        []float64{ 0.0,   0.0,   0.0,    0.0 }}

	pbar := matrix.FloatVector([]float64{.12, .10, .07, .03})
	S := matrix.FloatMatrixFromTable(Sdata)
	n := pbar.Rows()
	G := matrix.FloatDiagonal(n, -1.0)
	h := matrix.FloatZeros(n, 1)
	A := matrix.FloatWithValue(1, n, 1.0)
	b := matrix.FloatNew(1,1, []float64{1.0})

	var solopts cvx.SolverOptions
	solopts.MaxIter = 30
	solopts.ShowProgress = true

	mu := 1.0
	Smu := S.Copy().Scale(mu)
	pbarNeg := pbar.Copy().Scale(-1.0)
	fmt.Printf("Smu=\n%v\n", Smu.String())
	fmt.Printf("-pbar=\n%v\n", pbarNeg.String())

	sol, err := cvx.Qp(Smu, pbarNeg, G, h, A, b, &solopts, nil)

	fmt.Printf("status: %v\n", err)
	if sol != nil && sol.Status == cvx.Optimal {
		x := sol.Result.At("x")[0]
		ret := blas.DotFloat(x, pbar)
		risk := math.Sqrt(blas.DotFloat(x, S.Times(x)))
		fmt.Printf("ret=%.3f, risk=%.3f\n", ret, risk)
		fmt.Printf("x=\n%v\n", x)
	}
}

// Local Variables:
// tab-width: 4
// End:
