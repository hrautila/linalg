
package main

// Analytic centering example at the end of chapter 4 CVXOPT package documentation.
// This file is a rewrite of the original acent.py example.

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/linalg/lapack"
	"math"
	"fmt"
)

const (
	MAXITERS = 100
	ALPHA = 0.01
	BETA = 0.5
	TOL = 1e-8
)


// Computes analytic center of A*x <= b with A m by n of rank n. 
// We assume that b > 0 and the feasible set is bounded.
func Acent(A, b *matrix.FloatMatrix, niters int) (*matrix.FloatMatrix, []float64) {

	if niters <= 0 {
		niters = MAXITERS
	}
	ntdecrs := make([]float64, 0, niters)

	if A.Rows() != b.Rows() {
		return nil, nil
	}

	m, n := A.Size()
	x := matrix.FloatZeros(n, 1)
	H := matrix.FloatZeros(n, n)
	// Helper m*n matrix 
	Dmn := matrix.FloatZeros(m, n)

	for i := 0; i < niters; i++ {
		
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
			Dmn.SetColumnMatrix(i, d)
		}

		// Function mul creates element wise product of matrices.
		Asc := Dmn.Mul(A)
		blas.SyrkFloat(Asc, H, 1.0, 0.0, linalg.OptTrans)

        // Newton step is v = H^-1 * g.
		v := g.Copy().Neg()
		lapack.PosvFloat(H, v)

        // Directional derivative and Newton decrement.
		lam := blas.DotFloat(g, v)
		ntdecrs = append(ntdecrs, math.Sqrt(-lam))
		if ntdecrs[len(ntdecrs)-1] < TOL {
			fmt.Printf("last Newton decrement < TOL(%v)\n", TOL)
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
	// no solution !!
	fmt.Printf("Iteration %d exhausted\n", niters)
	return x, ntdecrs
}

func main() {
	// matrix string in row order presentation
	sA :=
		"[-7.44e-01  1.11e-01  1.29e+00  2.62e+00 -1.82e+00]" +
		"[ 4.59e-01  7.06e-01  3.16e-01 -1.06e-01  7.80e-01]" + 
		"[-2.95e-02 -2.22e-01 -2.07e-01 -9.11e-01 -3.92e-01]" +
		"[-7.75e-01  1.03e-01 -1.22e+00 -5.74e-01 -3.32e-01]" +
		"[-1.80e+00  1.24e+00 -2.61e+00 -9.31e-01 -6.38e-01]"
	
	sb :=
		"[ 8.38e-01]" +
		"[ 9.92e-01]" +
		"[ 9.56e-01]" +
		"[ 6.14e-01]" +
		"[ 6.56e-01]" +
		"[ 3.57e-01]" +
		"[ 6.36e-01]" +
		"[ 5.08e-01]" +
		"[ 8.81e-03]" +
		"[ 7.08e-02]"

	b, _ := matrix.FloatParsePy(sb)
	Al, _ := matrix.FloatParsePy(sA)
	Au := Al.Copy().Neg()
	A := matrix.FloatZeros(2*Al.Rows(), Al.Cols())
	A.SetSubMatrix(0, 0, Al)
	A.SetSubMatrix(Al.Rows(), 0, Au)

	//fmt.Printf("A (%d,%d):\n%v\n", A.Rows(), A.Cols(), A)
	//fmt.Printf("b (%d,%d):\n%v\n", b.Rows(), b.Cols(), b)
	x, nt := Acent(A, b, 10)
	fmt.Printf("nt:\n%v\n", nt)
	fmt.Printf("x :\n%v\n", x)
}


// Local Variables:
// tab-width: 4
// End:
