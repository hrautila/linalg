
package main

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/cvx"
	"fmt"
)

const (
	sS =
		"[ 3.57e+01]"+
		"[ 3.61e+01]"+
		"[ 2.29e+01]"+
		"[ 1.58e+01]"+
		"[-1.50e+01]"+
		"[ 1.22e+00]"+
		"[ 4.39e+01]"+
		"[ 2.99e-01]"+
		"[ 3.71e-01]"+
		"[-8.84e-01]"+
		"[ 1.07e+02]"+
		"[-2.80e+01]"+
		"[-2.24e+01]"+
		"[ 3.00e+01]"+
		"[ 1.22e+02]"+
		"[ 2.26e+01]"+
		"[ 1.90e+01]"+
		"[-2.30e+01]"+
		"[ 4.57e+01]"
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
	if h == nil {
	}

	//fmt.Printf("dims[l] = %v\n", dims.At("l"))
	//fmt.Printf("dims[q] = %v\n", dims.At("q"))
	//fmt.Printf("dims[s] = %v\n", dims.At("s"))
	//fmt.Printf("c [%d]\n", c.Rows())
	//sdot := cvx.Sdot(h, h, dims, 0)
	//fmt.Printf("sdot = %.3f\n", sdot)

	kkt, W := mksolver(G, A, dims)
	//doscale(G, W)
	//fmt.Printf("end of factoring:\ng=\n%v\n", kkt.Getg())
	//fmt.Printf("K=\n%v\nipiv=\n%v\n", kkt.GetK(), kkt.Getipiv())
	if kkt == nil {
	}
	if W == nil {
	}
	x, dy, s := mkprimal(kkt, W, c, b, h, dims)
	if x == nil || dy == nil || s == nil {
	}
	//fmt.Printf("** primal **\n")
	//fmt.Printf("x  [%d,%d]\n%v\n", x.Rows(), x.Cols(), x)
	//fmt.Printf("dy [%d,%d]\n%v\n", dy.Rows(), dy.Cols(), dy)
	//fmt.Printf("s  [%d,%d]\n%v\n", s.Rows(), s.Cols(), s)
	ts := cvx.MaxStep(s, dims, 0, nil)
	fmt.Printf("ts =%.4f\n", ts)
	//fmt.Printf("c [%d,%d]\n%v\n", c.Rows(), c.Cols(), c)
	//fmt.Printf("G [%d,%d]\n%v\n", G.Rows(), G.Cols(), G)
	//fmt.Printf("h [%d,%d]\n%v\n", h.Rows(), h.Cols(), h)
	//doGF(G, A, c, b, h, dims)

	var solopts cvx.SolverOptions
	solopts.MaxIter = 1
	solopts.ShowProgress = true
	cvx.ConeLp(c, G, h, nil, nil, dims, &solopts)
	//sinvtest(dims)
	//test_compute_scaling(dims)
	//test_jnrm2()
}

func doGF(G, A, c, b, h *matrix.FloatMatrix, dims *cvx.DimensionSet) {
	kkt, W := mksolver(G, A, dims)
	dx, y, z := mkdual(kkt, W, c, b, h, dims)
	if dx == nil || y == nil {
	}
	hrx := c.Copy()
	err := cvx.Sgemv(G, z, hrx, -1.0, 1.0, dims, linalg.OptTrans)
	fmt.Printf("err=%v\n", err)
	fmt.Printf("z=\n%v\n", z)
	fmt.Printf("hrx=\n%v\n", hrx)
}

func doscale(G *matrix.FloatMatrix, W *cvx.FloatMatrixSet) {
	g := matrix.FloatZeros(G.Rows(), 1)
	g.SetIndexes(matrix.MakeIndexSet(0, g.Rows(), 1),G.GetColumn(0, nil))
	fmt.Printf("** scaling g:\n%v\n", g)
	cvx.Scale(g, W, true, true)
	fmt.Printf("== scaled  g:\n%v\n", g)
}

func calcdims(dims *cvx.DimensionSet) (int, int, int) {
	cdim := dims.Sum("l", "q") + dims.SumSquared("s")
	cdim_pckd := dims.Sum("l", "q") + dims.SumPacked("s")
	cdim_diag := dims.Sum("l", "q", "s")
	return cdim, cdim_pckd, cdim_diag
}

func mkprimal(kktsolver cvx.KKT, W *cvx.FloatMatrixSet, c, b, h *matrix.FloatMatrix,
	dims *cvx.DimensionSet) (*matrix.FloatMatrix, *matrix.FloatMatrix, *matrix.FloatMatrix) {
	cdim, _, _ := calcdims(dims)
	x := c.Copy()
	x.Scale(0.0)
	dy := b.Copy()
	s := matrix.FloatZeros(cdim, 1)
	blas.CopyFloat(h, s)
	/*
	fmt.Printf("before solver\n")
	fmt.Printf("x  [%d,%d]\n%v\n", x.Rows(), x.Cols(), x)
	fmt.Printf("dy [%d,%d]\n%v\n", dy.Rows(), dy.Cols(), dy)
	fmt.Printf("s  [%d,%d]\n%v\n", s.Rows(), s.Cols(), s)
	 */
	kktsolver.Solve(x, dy, s)
	blas.ScalFloat(s, -1.0)
	return x, dy, s
}

func mkdual(kktsolver cvx.KKT, W *cvx.FloatMatrixSet, c, b, h *matrix.FloatMatrix,
	dims *cvx.DimensionSet) (*matrix.FloatMatrix, *matrix.FloatMatrix, *matrix.FloatMatrix) {
	cdim, _, _ := calcdims(dims)
	dx := c.Copy()
	dx.Scale(-1.0)
	y := b.Copy()
	y.Scale(0.0)
	z := matrix.FloatZeros(cdim, 1)
	/*
	fmt.Printf("before solver\n")
	fmt.Printf("x  [%d,%d]\n%v\n", x.Rows(), x.Cols(), x)
	fmt.Printf("dy [%d,%d]\n%v\n", dy.Rows(), dy.Cols(), dy)
	fmt.Printf("s  [%d,%d]\n%v\n", s.Rows(), s.Cols(), s)
	 */
	kktsolver.Solve(dx, y, z)
	return dx, y, z
}

func mksolver(G, A *matrix.FloatMatrix, dims *cvx.DimensionSet) (*cvx.KKTLdlSolver, *cvx.FloatMatrixSet) {
	kktsolver := cvx.CreateLdlSolver(G, dims, A, 0)
	W := cvx.FloatSetNew("d", "di", "v", "beta", "r", "rti")
	dd := dims.At("l")[0]
	mat := matrix.FloatOnes(dd, 1)
	W.Set("d", mat)
	mat = matrix.FloatOnes(dd, 1)
	W.Set("di", mat)
	dq := len(dims.At("q"))
	W.Set("beta", matrix.FloatOnes(dq, 1))

	for _, n := range dims.At("q")  {
		vm := matrix.FloatZeros(n, 1)
		vm.SetIndex(0, 1.0)
		W.Append("v", vm)
	}
	for _, n := range dims.At("s") {
		W.Append("r", matrix.FloatIdentity(n, n))
		W.Append("rti", matrix.FloatIdentity(n, n))
	}
	//f, err = kktsolver(W, nil, nil)
	_, err := kktsolver.Factor(W, nil, nil)
	if err != nil {
		fmt.Printf("kktsolver error: %s\n", err)
	}
	return kktsolver.(*cvx.KKTLdlSolver), W
}

func sinvtest(dims *cvx.DimensionSet) {
	sS2 :=
		"[ 4.58e+01]"+
		"[ 4.35e+01]"+
		"[ 2.83e+01]"+
		"[ 1.97e+01]"+
		"[-1.84e+01]"+
		"[ 2.69e+00]"+
		"[ 5.09e+01]"+
		"[ 7.07e-02]"+
		"[ 4.09e-01]"+
		"[-1.27e+00]"+
		"[ 1.75e+02]"+
		"[ 0.00e+00]"+
		"[ 0.00e+00]"+
		"[ 0.00e+00]"+
		"[ 1.04e+02]"+
		"[ 0.00e+00]"+
		"[ 0.00e+00]"+
		"[ 0.00e+00]"+
		"[ 3.89e+01]"
	lmbdaS2 :=
		"[ 6.77e+00]"+
		"[ 6.59e+00]"+
		"[ 4.27e+00]"+
		"[ 2.30e+00]"+
		"[-2.16e+00]"+
		"[ 3.15e-01]"+
		"[ 7.14e+00]"+
		"[ 4.95e-03]"+
		"[ 2.86e-02]"+
		"[-8.90e-02]"+
		"[ 1.32e+01]"+
		"[ 1.02e+01]"+
		"[ 6.24e+00]"+
		"[ 1.00e+00]"

	sinv_res :=
		"[-6.77e+00]"+
		"[-6.59e+00]"+
		"[-4.27e+00]"+
		"[-2.30e+00]"+
		"[ 2.16e+00]"+
		"[-3.15e-01]"+
		"[-7.14e+00]"+
		"[-4.95e-03]"+
		"[-2.86e-02]"+
		"[ 8.90e-02]"+
		"[-1.32e+01]"+
		"[-0.00e+00]"+
		"[-0.00e+00]"+
		"[-0.00e+00]"+
		"[-1.02e+01]"+
		"[-0.00e+00]"+
		"[-0.00e+00]"+
		"[-0.00e+00]"+
		"[-6.24e+00]"
		
	s, _ := matrix.FloatParsePy(sS2)
	lmbda, _ := matrix.FloatParsePy(lmbdaS2)
	cvx.Sinv(s, lmbda, dims, 0)
	blas.ScalFloat(s, -1.0)
	res_s, _ := matrix.FloatParsePy(sinv_res)
	fmt.Printf("res=\n%v\n", s)
	fmt.Printf("OK=%v\n", s.Equal(res_s))
}

func test_compute_scaling(dims *cvx.DimensionSet) {
	sS :=
		"[ 3.57e+01]"+
		"[ 3.61e+01]"+
		"[ 2.29e+01]"+
		"[ 1.58e+01]"+
		"[-1.50e+01]"+
		"[ 1.22e+00]"+
		"[ 4.39e+01]"+
		"[ 2.99e-01]"+
		"[ 3.71e-01]"+
		"[-8.84e-01]"+
		"[ 1.07e+02]"+
		"[-2.80e+01]"+
		"[-2.24e+01]"+
		"[ 3.00e+01]"+
		"[ 1.22e+02]"+
		"[ 2.26e+01]"+
		"[ 1.90e+01]"+
		"[-2.30e+01]"+
		"[ 4.57e+01]"

	zS :=
		"[ 1.28e+00]"+
		"[ 1.21e+00]"+
		"[ 1.23e+00]"+
		"[ 1.00e-02]"+
		"[ 7.99e-03]"+
		"[ 9.00e-02]"+
		"[ 1.16e+00]"+
		"[-6.30e-03]"+
		"[-4.97e-04]"+
		"[-5.57e-03]"+
		"[ 1.26e+00]"+
		"[ 8.42e-03]"+
		"[-3.01e-02]"+
		"[ 0.00e+00]"+
		"[ 1.13e+00]"+
		"[-8.81e-02]"+
		"[ 0.00e+00]"+
		"[ 0.00e+00]"+
		"[ 1.06e+00]"

	s, _ := matrix.FloatParsePy(sS)
	z, _ := matrix.FloatParsePy(zS)
	lmbda := matrix.FloatZeros(14,1)
	W, err := cvx.ComputeScaling(s, z, lmbda, dims, 0)
	fmt.Printf("lmbda=\n%v\n", lmbda)
	W.Print()
	if W == nil || err == nil {
	}
}

func test_jnrm2() {
	s, _ := matrix.FloatParsePy(sS)
	aa := cvx.Jnrm2(s, 4, 2)
	fmt.Printf("aa = %v\n", aa)
}

// Local Variables:
// tab-width: 4
// End:
