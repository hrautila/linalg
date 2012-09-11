
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

var xVal, sVal, zVal string
var spPath string
var maxIter int
var spVerbose bool

func init() {
	flag.BoolVar(&spVerbose, "V", false, "Savepoint verbose reporting.")
	flag.IntVar(&maxIter, "N", -1, "Max number of iterations.")
	flag.StringVar(&spPath, "sp", "", "savepoint directory")
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
	if len(spPath) > 0 {
		checkpnt.Reset(spPath)
		checkpnt.Activate()
		checkpnt.Verbose(spVerbose)
		checkpnt.Format("%.17f")
	}

	gdata := [][]float64{
		[]float64{ 16., 7.,  24.,  -8.,   8.,  -1.,  0., -1.,  0.,  0.,   7.,  
            -5.,   1.,  -5.,   1.,  -7.,   1.,   -7.,  -4.},
        []float64{-14., 2.,   7., -13., -18.,   3.,  0.,  0., -1.,  0.,   3.,  
            13.,  -6.,  13.,  12., -10.,  -6.,  -10., -28.},
        []float64{  5., 0., -15.,  12.,  -6.,  17.,  0.,  0.,  0., -1.,   9.,   
            6.,  -6.,   6.,  -7.,  -7.,  -6.,   -7., -11.}}
	
	hdata := []float64{-3., 5.,  12.,  -2., -14., -13., 10.,  0.,  0.,  0.,  68., 
        -30., -19., -30.,  99.,  23., -19.,   23.,  10.}



	c := matrix.FloatVector([]float64{-6., -4., -5.})
	G := matrix.FloatMatrixFromTable(gdata)
	h := matrix.FloatVector(hdata)

	dims := sets.DSetNew("l", "q", "s")
	dims.Set("l", []int{2})
	dims.Set("q", []int{4, 4})
	dims.Set("s", []int{3})
	

	var solopts cvx.SolverOptions
	solopts.MaxIter = 30
	if maxIter > 0 {
		solopts.MaxIter = maxIter
	}
	solopts.ShowProgress = true
	sol, err := cvx.ConeLp(c, G, h, nil, nil, dims, &solopts, nil, nil)
	if err == nil {
		x := sol.Result.At("x")[0]
		s := sol.Result.At("s")[0]
		z := sol.Result.At("z")[0]
		fmt.Printf("Optimal\n")
		fmt.Printf("x=\n%v\n", x.ToString("%.9f"))
		fmt.Printf("s=\n%v\n", s.ToString("%.9f"))
		fmt.Printf("z=\n%v\n", z.ToString("%.9f"))
		check(x, s, z)
	} else {
		fmt.Printf("status: %s\n", err)
	}
}



// Local Variables:
// tab-width: 4
// End:
