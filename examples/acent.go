
package main

// Analytic centering example at the end of chapter 4 CVXOPT package documentation.
// This file is a rewrite of the original acent.py example.

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/linalg/lapack"
)

const (
	MAXITERS := 100
	ALPHA := 0.01
	BETA := 0.5
	TOL := 1e-8
)

func Acent(A, b *matrix.FloatMatrix) (*matrix.FloatMatrix, []float64) {

	ntdecrs := make([]float64, 0, MAXITERS)

	m, n := A.Size()
	x := matrix.FloatZeros(n, 1)
	H := matrix.FloatZeros(n, n)
	// Helper m*n matrix 
	Dmn := matrix.FloatZeros(m, n)

	for i := 0; i < MAXITERS; i++ {
		
		// Gradient is g = A^T * (1.0/(b - A*x)). d = 1.0/(b - A*x)
		// d is m*1 matrix, g is n*1 matrix
		d := b.Minus(A.Times(x))
		d.Apply(nil, func(a float64)float64 { return 1.0/a })
		g := A.Transpose().Times(d)
		
        // Hessian is H = A^T * diag(1./(b-A*x))^2 * A.
		// in the original python code expression d[:,n*[0]] creates
		// a m*n matrix where each column is copy of column 0.
		// We do it here manually.
		for i := 0; i < n; i++ {
			Dmn.SetColumnVector(i, d)
		}
		// Function mul creates element wise product of matrices.
		Asc := Dmn.Mul(A)
		blas.SyrkFloat(Asc, H, linalg.OptTrans)

        // Newton step is v = H^-1 * g.
		v := g.Copy().Neg()
		lapack.PosvFloat(H, v)

        // Directional derivative and Newton decrement.
		lam := blas.DotFloat(g, v)
		ntdecrs.append(math.Sqrt(-lam))
		if ntdecrs[len(ntdecrs)-1] < TOL {
			return x, ntdecrs
		}

        // Backtracking line search.
		// y = d .* A*v
		y := d.Mul(A.Times(v))
		step := 1.0
		for 1 - step*y.Max() < 0 {
			step *= BETA
		}

	search:
		for ;; {
			// t = -step*y
			t := y.Copy().Scale(-step)
			// t = (1 + t) [e.g. t = 1 - step*y]
			t.Add(1.0)

			// ts = sum(log(1-step*y))
			ts := t.Log().Sum()
			if -ts < ALPHA*step*lam {
				break search
			}
			step *= BETA
		}
		v.Scale(step)
		x = x.Plus(v)
	}
}

func Main() {
}


// Local Variables:
// tab-width: 4
// End:
