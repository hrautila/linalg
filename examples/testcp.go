
package main


import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/cvx"
	"github.com/hrautila/go.opt/cvx/sets"
	"github.com/hrautila/go.opt/cvx/checkpnt"
	"fmt"
	"flag"
	"errors"
)

var xVal string
var spPath string
var spVerbose bool
var maxIter int

func init() {
	flag.BoolVar(&spVerbose, "V", false, "Savepoint verbose reporting.")
	flag.IntVar(&maxIter, "N", -1, "Max number of iterations.")
	flag.StringVar(&spPath, "sp", "", "savepoint directory")
	flag.StringVar(&xVal, "x", "", "Reference value for X")
}
	
func errorToRef(ref, val *matrix.FloatMatrix) (nrm float64, diff *matrix.FloatMatrix) {
	diff = ref.Minus(val)
	nrm = blas.Nrm2(diff).Float()
	return
}

func check(x *matrix.FloatMatrix) {
	if len(xVal) > 0 {
		ref, _ := matrix.FloatParseSpe(xVal)
		nrm, diff := errorToRef(ref, x)
		fmt.Printf("x: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
}


type acenterProg struct {
	rows, cols int
}


func (p *acenterProg) F0() (mnl int, x0 *matrix.FloatMatrix, err error) {
	err = nil
	mnl = 0
	x0 = matrix.FloatZeros(p.rows, p.cols)
	return 
}

func (p *acenterProg) F1(x *matrix.FloatMatrix)(f, Df *matrix.FloatMatrix, err error) {
	f = nil; Df = nil
	err = nil
	max := matrix.Abs(x).Max()
	//fmt.Printf("F1: max=%.3f x=\n%s\n", max, x)
	if max >= 1.0 {
		err = errors.New("max(abs(x)) >= 1.0")
		return
	}
	// u = 1 - x**2
	u := matrix.Pow(x, 2.0).Scale(-1.0).Add(1.0)
	val := - matrix.Log(u).Sum()
	f = matrix.FloatValue(val)
	Df = matrix.Div(matrix.Scale(x, 2.0), u).Transpose()
	return 
}

func (p *acenterProg) F2(x, z *matrix.FloatMatrix)(f, Df, H *matrix.FloatMatrix, err error) {
	f, Df, err = p.F1(x)
	u := matrix.Pow(x, 2.0).Scale(-1.0).Add(1.0)
	z0 := z.GetIndex(0)
	u2 := matrix.Pow(u, 2.0)
	hd := matrix.Div(matrix.Add(u2, 1.0), u2).Scale(2*z0)
	H = matrix.FloatDiagonal(hd.NumElements(), hd.FloatArray()...)
	return 
}

func acenter() *matrix.FloatMatrix {

	F := &acenterProg{3, 1}

	gdata := [][]float64{
		[]float64{0., -1.,  0.,  0., -21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.},
		[]float64{0.,  0., -1.,  0.,   0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.},
		[]float64{0.,  0.,  0., -1.,  -5.,   2., -17.,   2.,  -6.,   8., -17.,  -7., 6.}}

	G := matrix.FloatMatrixFromTable(gdata)
	h := matrix.FloatVector(
		[]float64{1.0, 0.0, 0.0, 0.0, 20., 10., 40., 10., 80., 10., 40., 10., 15.})
	
	var solopts cvx.SolverOptions
	solopts.MaxIter = 2
	if maxIter > -1 {
		solopts.MaxIter = maxIter
	}
	solopts.ShowProgress = true

	dims := sets.NewDimensionSet("l", "q", "s")
	dims.Set("l", []int{0})
	dims.Set("q", []int{4})
	dims.Set("s", []int{3})

	var err error
	var sol *cvx.Solution

	sol, err = cvx.Cp(F, G, h, nil, nil, dims, &solopts)
	if err == nil && sol.Status == cvx.Optimal {
		return sol.Result.At("x")[0]
	} else {
		fmt.Printf("result: %v\n", err)
	}
	return nil
}

func main() {
	flag.Parse()
	if len(spPath) > 0 {
		checkpnt.Reset(spPath)
		checkpnt.Activate()
		checkpnt.Verbose(spVerbose)
		checkpnt.Format("%.7f")
	}
	
	x := acenter()
	if x != nil {
		fmt.Printf("x = \n%v\n", x.ToString("%.5f"))
		check(x)
	}
}

// Local Variables:
// tab-width: 4
// End:
