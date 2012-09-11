
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
	if len(xVal) > 0 {
		ref, _ := matrix.FloatParseSpe(xVal)
		nrm, diff := error(ref, x)
		fmt.Printf("x: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
	if len(sVal) > 0 {
		ref, _ := matrix.FloatParseSpe(sVal)
		nrm, diff := error(ref, s)
		fmt.Printf("s: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
	if len(zVal) > 0 {
		ref, _ := matrix.FloatParseSpe(zVal)
		nrm, diff := error(ref, z)
		fmt.Printf("z: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
}

func main() {
	flag.Parse()
	reftest := flag.NFlag() > 0

	gdata := [][]float64{
		[]float64{ 2.0, 1.0, -1.0,  0.0 },
		[]float64{ 1.0, 2.0,  0.0, -1.0 }}

	c := matrix.FloatVector([]float64{-4.0, -5.0})
	G := matrix.FloatMatrixFromTable(gdata, matrix.ColumnOrder)
	h := matrix.FloatVector([]float64{3.0, 3.0, 0.0, 0.0})

	var solopts cvx.SolverOptions
	solopts.MaxIter = 30
	solopts.ShowProgress = true
	sol, err := cvx.Lp(c, G, h, nil, nil, &solopts, nil, nil)
	if sol != nil && sol.Status == cvx.Optimal {
		x := sol.Result.At("x")[0]
		s := sol.Result.At("s")[0]
		z := sol.Result.At("z")[0]
		fmt.Printf("x=\n%v\n", x.ToString("%.9f"))
		fmt.Printf("s=\n%v\n", s.ToString("%.9f"))
		fmt.Printf("z=\n%v\n", z.ToString("%.9f"))
		if reftest {
			check(x, s, z)
		}
	} else {
		fmt.Printf("status: %v\n", err)
	}
}

// Local Variables:
// tab-width: 4
// End:
