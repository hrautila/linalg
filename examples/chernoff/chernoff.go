//
// cvxopt/examples/book/chap7/chernoff.py 
//
package main

import (
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/cvx"
	"code.google.com/p/plotinum/vg"
	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"
	"image/color"
	"math"
	"fmt"
)

func dataset(xs, ys []float64) plotter.XYs {
	pts := make(plotter.XYs, len(xs))
	for i := range xs {
		pts[i].X = xs[i]
		pts[i].Y = ys[i]
	}
	return pts
}

func plotData(name string, xs, ys []float64) {
	p, err := plot.New()
	if err != nil {
		fmt.Printf("Cannot create new plot: %s\n", err)
		return
	}
	p.Title.Text = "Chernoff lower bound"
	p.X.Label.Text = "Sigma"
	p.X.Min = 0.2
	p.X.Max = 0.5
	p.Y.Label.Text = "Probability of correct detection"
	p.Y.Min = 0.9
	p.Y.Max = 1.0
	p.Add(plotter.NewGrid())

	l := plotter.NewLine(dataset(xs, ys))
	l.LineStyle.Width = vg.Points(1)
	//l.LineStyle.Dashes = []vg.Length(vg.Points(5), vg.Points(5))
	l.LineStyle.Color = color.RGBA{B:255, A:255}

	p.Add(l)
	if err := p.Save(4, 4, name); err != nil {
		fmt.Printf("Save to '%s' failed: %s\n", name, err)
	}
}

func main() {
	m := 6
	Vdata := [][]float64{
		[]float64{1.0, -1.0, -2.0, -2.0, 0.0, 1.5, 1.0},
		[]float64{1.0, 2.0, 1.0, -1.0, -2.0, -1.0, 1.0}}

	V := matrix.FloatMatrixFromTable(Vdata, matrix.RowOrder)

	// V[1, :m] - V[1,1:]
	a0 := matrix.Minus(V.GetSubMatrix(1, 0, 1, m), V.GetSubMatrix(1, 1, 1))
	// V[0, :m] - V[0,1:]
	a1 := matrix.Minus(V.GetSubMatrix(0, 0, 1, m), V.GetSubMatrix(0, 1, 1))
	A0, _ := matrix.FloatMatrixStacked(matrix.StackDown, a0.Scale(-1.0), a1)
	A0 = A0.Transpose()
	b0 := matrix.Mul(A0, V.GetSubMatrix(0, 0, 2, m).Transpose())
	b0 = matrix.Times(b0, matrix.FloatWithValue(2, 1, 1.0))

	A := make([]*matrix.FloatMatrix, 0)
	b := make([]*matrix.FloatMatrix, 0)
	A = append(A, A0)
	b = append(b, b0)

	// List of symbols
	C := make([]*matrix.FloatMatrix, 0)
	C = append(C, matrix.FloatZeros(2, 1))
	var row *matrix.FloatMatrix = nil
	for k := 0; k < m; k++ {
		row = A0.GetRow(k, row)
		nrm := blas.Nrm2Float(row)
		row.Scale(2.0 * b0.GetIndex(k) / (nrm * nrm))
		C = append(C, row.Transpose())
	}

	// Voronoi set around C[1]
	A1 := matrix.FloatZeros(3,2)
	A1.SetSubMatrix(0, 0, A0.GetSubMatrix(0, 0, 1).Scale(-1.0))
	A1.SetSubMatrix(1, 0, matrix.Minus(C[m], C[1]).Transpose())
	A1.SetSubMatrix(2, 0, matrix.Minus(C[2], C[1]).Transpose())
	b1 := matrix.FloatZeros(3,1)
	b1.SetIndex(0, -b0.GetIndex(0))
	v := matrix.Times(A1.GetRow(1, nil), matrix.Plus(C[m], C[1])).Float() * 0.5
	b1.SetIndex(1, v)
	v = matrix.Times(A1.GetRow(2, nil), matrix.Plus(C[2], C[1])).Float() * 0.5
	b1.SetIndex(2, v)
	A = append(A, A1)
	b = append(b, b1)

	// Voronoi set around C[2] ... C[5]
	for k := 2; k < 6; k++ {
		A1 = matrix.FloatZeros(3,2)
		A1.SetSubMatrix(0, 0, A0.GetSubMatrix(k-1, 0, 1).Scale(-1.0))
		A1.SetSubMatrix(1, 0, matrix.Minus(C[k-1], C[k]).Transpose())
		A1.SetSubMatrix(2, 0, matrix.Minus(C[k+1], C[k]).Transpose())
		b1 = matrix.FloatZeros(3,1)
		b1.SetIndex(0, -b0.GetIndex(k-1))
		v := matrix.Times(A1.GetRow(1, nil), matrix.Plus(C[k-1], C[k])).Float() * 0.5
		b1.SetIndex(1, v)
		v = matrix.Times(A1.GetRow(2, nil), matrix.Plus(C[k+1], C[k])).Float() * 0.5
		b1.SetIndex(2, v)
		A = append(A, A1)
		b = append(b, b1)
	}

	// Voronoi set around C[6]
	A1 = matrix.FloatZeros(3,2)
	A1.SetSubMatrix(0, 0, A0.GetSubMatrix(5, 0, 1).Scale(-1.0))
	A1.SetSubMatrix(1, 0, matrix.Minus(C[1], C[6]).Transpose())
	A1.SetSubMatrix(2, 0, matrix.Minus(C[5], C[6]).Transpose())
	b1 = matrix.FloatZeros(3,1)
	b1.SetIndex(0, -b0.GetIndex(5))
	v = matrix.Times(A1.GetRow(1, nil), matrix.Plus(C[1], C[6])).Float() * 0.5
	b1.SetIndex(1, v)
	v = matrix.Times(A1.GetRow(2, nil), matrix.Plus(C[5], C[6])).Float() * 0.5
	b1.SetIndex(2, v)

	A = append(A, A1)
	b = append(b, b1)

	P := matrix.FloatIdentity(2)
	q := matrix.FloatZeros(2, 1)
	solopts := &cvx.SolverOptions{ShowProgress:false, MaxIter:30}
	ovals := make([]float64, 0)
	for k := 1; k < 7; k++ {
		sol, err := cvx.Qp(P, q, A[k], b[k], nil, nil, solopts, nil)
		_ = err
		x := sol.Result.At("x")[0]
		ovals = append(ovals, math.Pow(blas.Nrm2Float(x), 2.0))
	}

	optvals := matrix.FloatVector(ovals)
	fmt.Printf("optvals=\n%v\n", optvals)
	
	rangeFunc := func(n int)[]float64 {
		r := make([]float64, 0)
		for i := 0; i < n; i++ {
			r = append(r, float64(i))
		}
		return r
	}

	nopts := 200
	sigmas := matrix.FloatVector(rangeFunc(nopts))
	sigmas.Scale((0.5 - 0.2)/float64(nopts)).Add(0.2)

	bndsVal := func(sigma float64)float64 {
		// 1.0 - sum(exp( -optvals/(2*sigma**2)))
		return 1.0 - matrix.Exp(matrix.Scale(optvals, -1.0/(2*sigma*sigma))).Sum()
	}
	bnds := matrix.FloatZeros(sigmas.NumElements(), 1)
	for j, v := range sigmas.FloatArray() {
		bnds.SetIndex(j, bndsVal(v))
	}
	plotData("plot.png", sigmas.FloatArray(), bnds.FloatArray())
}

// Local Variables:
// tab-width: 4
// End:
