
package main

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/cvx"
	"fmt"
	"flag"
)

var xVal, sq0Val, sq1Val, zq0Val, zq1Val string

func init() {
	flag.StringVar(&xVal, "x", "", "Reference value for X")
	flag.StringVar(&sq0Val, "sq0", "", "Reference value for SQ[0]")
	flag.StringVar(&sq1Val, "sq1", "", "Reference value for SQ[1]")
	flag.StringVar(&zq0Val, "zq0", "", "Reference value for ZQ[0]")
	flag.StringVar(&zq1Val, "zq1", "", "Reference value for ZQ[1]")
}
	
func error(ref, val *matrix.FloatMatrix) (nrm float64, diff *matrix.FloatMatrix) {
	diff = ref.Minus(val)
	nrm = blas.Nrm2(diff).Float()
	return
}

func check(x, sq0, sq1, zq0, zq1 *matrix.FloatMatrix) {
	if len(xVal) > 0 {
		ref, _ := matrix.FloatParseSpe(xVal)
		nrm, diff := error(ref, x)
		fmt.Printf("x: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
	if len(sq0Val) > 0 {
		ref, _ := matrix.FloatParseSpe(sq0Val)
		nrm, diff := error(ref,sq0)
		fmt.Printf("sq0: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
	if len(sq1Val) > 0 {
		ref, _ := matrix.FloatParseSpe(sq1Val)
		nrm, diff := error(ref,sq1)
		fmt.Printf("sq1: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
	if len(zq0Val) > 0 {
		ref, _ := matrix.FloatParseSpe(zq0Val)
		nrm, diff := error(ref, zq0)
		fmt.Printf("zq0: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
	if len(zq1Val) > 0 {
		ref, _ := matrix.FloatParseSpe(zq1Val)
		nrm, diff := error(ref, zq1)
		fmt.Printf("zq1: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
}

func main() {
	flag.Parse()
	reftest := flag.NFlag() > 0
	
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
		if reftest {
			sq0 := sol.Result.At("sq")[0]
			sq1 := sol.Result.At("sq")[1]
			zq0 := sol.Result.At("zq")[0]
			zq1 := sol.Result.At("zq")[1]
			check(x, sq0, sq1, zq0, zq1)
		}
	}
}

// Local Variables:
// tab-width: 4
// End:
