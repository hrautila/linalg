
package main

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/linalg/blas"
	//"github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/cvx"
	"fmt"
)


func main() {
	sc :=
		"[-6.00e+00]"+
		"[-4.00e+00]"+
		"[-5.00e+00]"

	sG :=
		"[ 1.60e+01 -1.40e+01  5.00e+00]"+
		"[ 7.00e+00  2.00e+00  0.00e+00]"+
		"[ 2.40e+01  7.00e+00 -1.50e+01]"+
		"[-8.00e+00 -1.30e+01  1.20e+01]"+
		"[ 8.00e+00 -1.80e+01 -6.00e+00]"+
		"[-1.00e+00  3.00e+00  1.70e+01]"+
		"[ 0.00e+00  0.00e+00  0.00e+00]"+
		"[-1.00e+00  0.00e+00  0.00e+00]"+
		"[ 0.00e+00 -1.00e+00  0.00e+00]"+
		"[ 0.00e+00  0.00e+00 -1.00e+00]"+
		"[ 7.00e+00  3.00e+00  9.00e+00]"+
		"[-5.00e+00  1.30e+01  6.00e+00]"+
		"[ 1.00e+00 -6.00e+00 -6.00e+00]"+
		"[-5.00e+00  1.30e+01  6.00e+00]"+
		"[ 1.00e+00  1.20e+01 -7.00e+00]"+
		"[-7.00e+00 -1.00e+01 -7.00e+00]"+
		"[ 1.00e+00 -6.00e+00 -6.00e+00]"+
		"[-7.00e+00 -1.00e+01 -7.00e+00]"+
		"[-4.00e+00 -2.80e+01 -1.10e+01]"

	sh :=
		"[-3.00e+00]"+
		"[ 5.00e+00]"+
		"[ 1.20e+01]"+
		"[-2.00e+00]"+
		"[-1.40e+01]"+
		"[-1.30e+01]"+
		"[ 1.00e+01]"+
		"[ 0.00e+00]"+
		"[ 0.00e+00]"+
		"[ 0.00e+00]"+
		"[ 6.80e+01]"+
		"[-3.00e+01]"+
		"[-1.90e+01]"+
		"[-3.00e+01]"+
		"[ 9.90e+01]"+
		"[ 2.30e+01]"+
		"[-1.90e+01]"+
		"[ 2.30e+01]"+
		"[ 1.00e+01]"


	c,_ := matrix.FloatParsePy(sc)
	G,_ := matrix.FloatParsePy(sG)
	h,_ := matrix.FloatParsePy(sh)
	A := matrix.FloatZeros(0, c.Rows())
	b := matrix.FloatZeros(0, 1)

	dims := cvx.DSetNew("l", "q", "s")
	dims.Set("l", []int{2})
	dims.Set("q", []int{4, 4})
	dims.Set("s", []int{3})
	if b == nil {
	}
	if c == nil {
	}
	if h == nil || A == nil {
	}
	

	var solopts cvx.SolverOptions
	solopts.MaxIter = 30
	solopts.ShowProgress = true
	sol, err := cvx.ConeLp(c, G, h, nil, nil, dims, &solopts, nil, nil)
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
