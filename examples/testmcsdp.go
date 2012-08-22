
package main

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/linalg/lapack"
	"github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/cvx"
	"fmt"
	"flag"
)

var xVal, zVal, dataVal string

func init() {
	flag.StringVar(&xVal, "x", "", "Reference value for X")
	flag.StringVar(&zVal, "z", "", "Reference value for Z")
	flag.StringVar(&dataVal, "data", "", "Problem data")
}
	
func errorToRef(ref, val *matrix.FloatMatrix) (nrm float64, diff *matrix.FloatMatrix) {
	diff = ref.Minus(val)
	nrm = blas.Nrm2(diff).Float()
	return
}

func check(x, z *matrix.FloatMatrix) {
	if len(xVal) > 0 {
		ref, _ := matrix.FloatParseSpe(xVal)
		nrm, diff := errorToRef(ref, x)
		fmt.Printf("x: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
	if len(zVal) > 0 {
		ref, _ := matrix.FloatParseSpe(zVal)
		nrm, diff := errorToRef(ref, z)
		fmt.Printf("z: nrm=%.9f\n", nrm)
		if nrm > 10e-7 {
			fmt.Printf("diff=\n%v\n", diff.ToString("%.12f"))
		}
	}
}


// Internal type for MatrixG interface
type matrixFs struct {
	n int
}

func (g *matrixFs)Gf(x, y *matrix.FloatMatrix, alpha, beta float64, trans linalg.Option)  error {
	//
	// y := alpha*(-diag(x)) + beta*y
	//
	if linalg.Equal(trans, linalg.OptNoTrans) {
		blas.ScalFloat(y, beta)
		blas.AxpyFloat(x, y, -alpha, &linalg.IOpt{"incy", g.n+1})
	} else {
		blas.ScalFloat(y, beta)
		blas.AxpyFloat(x, y, -alpha, &linalg.IOpt{"incx", g.n+1})
	}
	return nil
}

func mcsdp(w *matrix.FloatMatrix) (*cvx.Solution, error) {
	//
    // Returns solution x, z to 
    //
    //    (primal)  minimize    sum(x)
    //              subject to  w + diag(x) >= 0
    //
    //    (dual)    maximize    -tr(w*z)
    //              subject to  diag(z) = 1
    //                          z >= 0.
	//
	n := w.Rows()
	G := &matrixFs{n}

	cngrnc := func(r, x *matrix.FloatMatrix, alpha float64) (err error) {
		// Congruence transformation
		//
		//    x := alpha * r'*x*r.
		//
		// r and x are square matrices.  
		//

		err = nil
		// scale diagonal of x by 1/2
		x.Scale(0.5, matrix.MakeDiagonalSet(n, n)...)

		// a := tril(x)*r
		// (python: a = +r is really making a copy of r)
		a := r.Copy()

		// tx = matrix(x, (n,n)) is copying and reshaping
		tx := x.Copy()
		matrix.Reshape(tx, n, n)

		err = blas.TrmmFloat(tx, a, 1.0, linalg.OptLeft)
		
        // x := alpha*(a*r' + r*a') 
		err = blas.Syr2kFloat(r, a, tx, alpha, 0.0, linalg.OptTrans)
		// x[:] = tx[:] 
		matrix.Set(x, tx)
		return 
	}

	Fkkt := func(W *cvx.FloatMatrixSet) (cvx.KKTFunc, error) {

        //    Solve
        //                  -diag(z)                           = bx
        //        -diag(x) - inv(rti*rti') * z * inv(rti*rti') = bs
        //
        //    On entry, x and z contain bx and bs.  
        //    On exit, they contain the solution, with z scaled
        //    (inv(rti)'*z*inv(rti) is returned instead of z).
        //
        //    We first solve 
        //
        //        ((rti*rti') .* (rti*rti')) * x = bx - diag(t*bs*t) 
        //
        //    and take z  = -rti' * (diag(x) + bs) * rti.

		var err error = nil
		rti := W.At("rti")[0]

        // t = rti*rti' as a nonsymmetric matrix.
		t := matrix.FloatZeros(n, n)
		err = blas.GemmFloat(rti, rti, t, 1.0, 0.0, linalg.OptTransB)
		if err != nil {	return nil, err	} 
		
        // Cholesky factorization of tsq = t.*t.
		tsq := t.Mul(t) 
		err = lapack.Potrf(tsq)
		if err != nil {	return nil, err	} 
		
		f := func(x, y, z *matrix.FloatMatrix) (err error) {
            // tbst := t * zs * t = t * bs * t
			tbst := z.Copy()
			matrix.Reshape(tbst, n, n)
			cngrnc(t, tbst, 1.0)
			
            // x := x - diag(tbst) = bx - diag(rti*rti' * bs * rti*rti')
			diag := matrix.FloatVector(tbst.Get(matrix.MakeDiagonalSet(n, n)...))
			matrix.Set(x, x.Minus(diag))

            // x := (t.*t)^{-1} * x = (t.*t)^{-1} * (bx - diag(t*bs*t))
			err = lapack.Potrs(tsq, x)
            // z := z + diag(x) = bs + diag(x)
			z.AddIndexes(matrix.MakeIndexSet(0, n*n, n+1), x.FloatArray())

            // z := -rti' * z * rti = -rti' * (diag(x) + bs) * rti 
			cngrnc(rti, z, -1.0)
			return nil
		}
		return f, nil
	}
	
	c := matrix.FloatWithValue(n, 1, 1.0)

	// initial feasible x: x = 1.0 - min(lmbda(w))
	lmbda := matrix.FloatZeros(n, 1)
	wp := w.Copy()
	lapack.Syevx(wp, lmbda, nil, 0.0, nil, []int{1,1}, linalg.OptRangeInt)
	x0 := matrix.FloatZeros(n, 1).Add(-lmbda.GetAt(0,0)+1.0)
	s0 := w.Copy()
	s0.AddIndexes(matrix.MakeDiagonalSet(n, n), x0.FloatArray())
	matrix.Reshape(s0, n*n, 1)
	
	// initial feasible z is identity
	z0 := matrix.FloatIdentity(n)
	matrix.Reshape(z0, n*n, 1)
	
	dims := cvx.DSetNew("l", "q", "s")
	dims.Set("s", []int{n})

	primalstart := cvx.FloatSetNew("x", "s")
	dualstart := cvx.FloatSetNew("z")
	primalstart.Set("x", x0)
	primalstart.Set("s", s0)
	dualstart.Set("z", z0)

	var solopts cvx.SolverOptions
	solopts.MaxIter = 30
	solopts.ShowProgress = true
	h := w.Copy()
	matrix.Reshape(h, h.NumElements(), 1)
	return cvx.ConeLpCustom(c, G, h, nil, nil, dims, Fkkt, &solopts, primalstart, dualstart)
}

func main() {
	var data *matrix.FloatMatrix = nil
	flag.Parse()
	dataCount := 0
	if len(dataVal) > 0 {
		dataCount = 1
		data, _ = matrix.FloatParseSpe(dataVal)
		if data == nil {
			fmt.Printf("could not parse:\n%s\n", dataVal)
			return
		}
	} else {
		data = matrix.FloatRandom(20, 20, false)
	}
	reftest := flag.NFlag() - dataCount > 0

	sol, err := mcsdp(data)
	if sol != nil && sol.Status == cvx.Optimal {
		x := sol.Result.At("x")[0]
		z := sol.Result.At("z")[0]
		matrix.Reshape(z, data.Rows(), data.Rows())
		fmt.Printf("x=\n%v\n", x.ToString("%.9f"))
		fmt.Printf("z=\n%v\n", z.ToString("%.9f"))
		if reftest {
			check(x, z)
		}
	} else {
		fmt.Printf("status: %v\n", err)
	}
}

// Local Variables:
// tab-width: 4
// End:
