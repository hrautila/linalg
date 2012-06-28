
// Analytic centering example at the end of chapter 4 CVXOPT package documentation.
// This file is a rewrite of the original acent.py example.

func Acent(A, b *matrix.FloatMatrix) (*matrix.FloatMatrix, []int) {
	MAXITERS := 100
	ALPHA := 0.01
	BETA := 0.5
	TOL := 1e-8

	m, n := A.Size()
	x := matrix.FloatZeros(n, 1)
	H := matrix.FloatZeros(n, n)

	for i := 0; i < MAXITERS; i++ {
		
		// Gradient is g = A^T * (1.0/(b - A*x)). d = 1.0/(b - A*x)
		d := b.Minus(A.Times(x))
		d.Apply(nil, func(a float64)float64 { return 1.0/a })
		g := A.Transpose().Times(d)
	}
}


// Local Variables:
// tab-width: 4
// End:
