
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

var xVal, zVal, AVal, bVal string

func init() {
	flag.StringVar(&xVal, "x", "", "Reference value for X")
	flag.StringVar(&zVal, "z", "", "Reference value for Z")
	flag.StringVar(&AVal, "A", "", "Problem data A")
	flag.StringVar(&bVal, "b", "", "Problem data b")
}
	
func errorToRef(ref, val *matrix.FloatMatrix) (nrm float64, diff *matrix.FloatMatrix) {
	diff = matrix.Minus(ref, val)
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
	A *matrix.FloatMatrix
}

func (g *matrixFs)Gf(x, y *matrix.FloatMatrix, alpha, beta float64, trans linalg.Option)  error {
	//

	//fmt.Printf("Gf:x=\n%v\n", x.ToString("%.5f"))
	//fmt.Printf("Gf:y=\n%v\n", y.ToString("%.5f"))
	m, n := g.A.Size()
	y.Scale(beta)

	// x_n = x[:n]
	x_n := matrix.FloatVector(x.FloatArray()[:n])
	
	// x_n_2n = x[n:2*n]
	x_n_2n := matrix.FloatVector(x.FloatArray()[n:2*n])

	if linalg.Equal(trans, linalg.OptNoTrans) {
		// y += alpha * G * x

		// y[:n] += alpha * (x[:n] - x[n:2*n])
		y_n := matrix.Minus(x_n, x_n_2n).Scale(alpha)
		y.AddIndexes(matrix.Indexes(n), y_n.FloatArray())

		// y[n:2*n] += alpha * (-x[:n] - x[n:2*n]) = -alpha * (x[:n]+x[n:2*n])
		y_n = matrix.Plus(x_n, x_n_2n).Scale(-alpha)
		y.AddIndexes(matrix.Indexes(n, 2*n), y_n.FloatArray())

		// y[2*n+1:] += -alpha * A * x[:n]
		y_2n :=	matrix.Times(g.A, x_n).Scale(-alpha)
		y.AddIndexes(matrix.Indexes(2*n+1, y.NumElements()), y_2n.FloatArray())
	} else {
		// x_m = x[-m:]
		x_m := matrix.FloatVector(x.FloatArray()[x.NumElements()-m:])

		// x_tmp = (x[:n] - x[n:2*n] - A.T * x[-m:])
		x_tmp := matrix.Minus(x_n, x_n_2n, matrix.Times(g.A.Transpose(), x_m))

		// y[:n] += alpha * (x[:n] - x[n:2*n] - A.T * x[-m:])
		y.AddIndexes(matrix.Indexes(n), x_tmp.Scale(alpha).FloatArray())

		x_tmp = matrix.Plus(x_n, x_n_2n).Scale(-alpha)
		y.AddIndexes(matrix.Indexes(n, y.NumElements()), x_tmp.FloatArray())
	}
	//fmt.Printf("end Gf:x=\n%v\n", x.ToString("%.5f"))
	//fmt.Printf("end Gf:y=\n%v\n", y.ToString("%.5f"))
	return nil
}

func qcl1(A, b *matrix.FloatMatrix) (*cvx.Solution, error) {

     // Returns the solution u, z of
     //
     //   (primal)  minimize    || u ||_1       
     //             subject to  || A * u - b ||_2  <= 1
     //
     //   (dual)    maximize    b^T z - ||z||_2
     //             subject to  || A'*z ||_inf <= 1.
	 //
     // Exploits structure, assuming A is m by n with m >= n. 

	m, n := A.Size()
	Fkkt := func(W *cvx.FloatMatrixSet) (f cvx.KKTFunc, err error) {

		err = nil
		f = nil
		beta := W.At("beta")[0].GetIndex(0)
		v := W.At("v")[0]

		// As = 2 * v *(v[1:].T * A)
		v_1 := matrix.FloatNew(1, v.NumElements()-1, v.FloatArray()[1:])
		//fmt.Printf("vt=\n%v\n", vt.ToString("%.5f"))
		As := matrix.Times(v, matrix.Times(v_1, A)).Scale(2.0)

		As_1 := As.GetSubMatrix(1, 0, m, n)
		As_1.Scale(-1.0)
		As.SetSubMatrix(1, 0, matrix.Minus(As_1, A))
		As.Scale(1.0/beta)
		//fmt.Printf("As=\n%v\n", As.ToString("%.5f"))

		S := matrix.Times(As.Transpose(), As)
		//fmt.Printf("S=\n%v\n", S.ToString("%.12f"))
		d1 := matrix.FloatVector(W.At("d")[0].FloatArray()[:n])
		d2 := matrix.FloatVector(W.At("d")[0].FloatArray()[n:])
		fmt.Printf("d1=\n%v\n", d1.ToString("%.17f"))
		fmt.Printf("d2=\n%v\n", d2.ToString("%.17f"))

		// D = 4.0 * (d1**2 + d2**2)**-1
		d := matrix.Plus(matrix.Mul(d1,d1), matrix.Mul(d2, d2)).Inv().Scale(4.0)
		//fmt.Printf("d=\n%v\n", d.ToString("%.12f"))
		// S[::n+1] += d
		S.AddIndexes(matrix.DiagonalIndexes(S), d.FloatArray())

		//fmt.Printf("S=\n%v\n", S.ToString("%.12f"))
		err = lapack.Potrf(S)
		//fmt.Printf("potrf S=\n%v\n", S.ToString("%.12f"))
		if err != nil {	return }
		
		f = func(x, y, z *matrix.FloatMatrix) (err error) {
            // -- z := - W**-T * z 

			//fmt.Printf("f start : x=\n%v\n", x.ToString("%.5f"))
			//fmt.Printf("f start : z=\n%v\n", z.ToString("%.5f"))

			// z[:n] = -div( z[:n], d1 )
			z_val := matrix.FloatVector(z.FloatArray()[:n])
			z_res := matrix.Div(z_val, d1).Scale(-1.0)
			z.SetIndexes(matrix.Indexes(n), z_res.FloatArray())

			// z[n:2*n] = -div( z[n:2*n], d2 )
			z_val = matrix.FloatVector(z.FloatArray()[n:2*n])
			z_res = matrix.Div(z_val, d2).Scale(-1.0)
			z.SetIndexes(matrix.Indexes(n, 2*n), z_res.FloatArray())
			
			// z[2*n:] -= 2.0*v*( v[0]*z[2*n] - blas.dot(v[1:], z[2*n+1:]) ) 
			v0_z2n := v.GetIndex(0)*z.GetIndex(2*n)
			v1_z2n := blas.DotFloat(v, z, &linalg.IOpt{"offsetx", 1}, &linalg.IOpt{"offsety", 2*n+1})
			z_res = matrix.Scale(v, -2.0*(v0_z2n - v1_z2n))
			z.AddIndexes(matrix.Indexes(2*n, z.NumElements()), z_res.FloatArray())

			// z[2*n+1:] *= -1.0
			z.Scale(-1.0, matrix.Indexes(2*n+1, z.NumElements())...)

			// z[2*n:] /= beta
			z.Scale(1.0/beta, matrix.Indexes(2*n, z.NumElements())...)

			//fmt.Printf("f 0: z=\n%v\n", z.ToString("%.5f"))

			// -- x := x - G' * W**-1 * z

			// z_n = z[:n], z_2n = z[n:2*n], z_m = z[-(m+1):], 
			z_n := matrix.FloatVector(z.FloatArray()[:n])
			z_2n := matrix.FloatVector(z.FloatArray()[n:2*n])
			z_m := matrix.FloatVector(z.FloatArray()[z.NumElements()-(m+1):])

			// x[:n] -= div(z[:n], d1) - div(z[n:2*n], d2) + As.T * z[-(m+1):]
			z_res = matrix.Minus(matrix.Div(z_n, d1), matrix.Div(z_2n, d2))
			a_res := matrix.Times(As.Transpose(), z_m)
			z_res = matrix.Plus(z_res, a_res).Scale(-1.0)
			x.AddIndexes(matrix.Indexes(n), z_res.FloatArray())

			// x[n:] += div(z[:n], d1) + div(z[n:2*n], d2) 
			z_res = matrix.Plus(matrix.Div(z_n, d1), matrix.Div(z_2n, d2))
			z_res = matrix.Div(z_res, d2)
			x.AddIndexes(matrix.Indexes(n, x.NumElements()), z_res.FloatArray())
			
            // Solve for x[:n]:
            //
            //    S*x[:n] = x[:n] - (W1**2 - W2**2)(W1**2 + W2**2)^-1 * x[n:]

			// w1 = (d1**2 - d2**2), w2 = (d1**2 + d2**2)
            w1 := matrix.Minus(matrix.Mul(d1, d1), matrix.Mul(d2, d2))
            w2 := matrix.Plus(matrix.Mul(d1, d1), matrix.Mul(d2, d2))

			// x[:n] = -mul( div(w1, w2), x[n:])
			x_n := matrix.FloatVector(x.FloatArray()[n:])
			x_val := matrix.Mul(matrix.Div(w1, w2), x_n).Scale(-1.0)
			x.AddIndexes(matrix.Indexes(n), x_val.FloatArray())

            // Solve for x[n:]:
            //
            //    (d1**-2 + d2**-2) * x[n:] = x[n:] + (d1**-2 - d2**-2)*x[:n]

			//fmt.Printf("f potrs: x=\n%v\n", x.ToString("%.5f"))
			err = lapack.Potrs(S, x)

            // Solve for x[n:]:
            //
            //    (d1**-2 + d2**-2) * x[n:] = x[n:] + (d1**-2 - d2**-2)*x[:n]

			// w1 = (d1**-2 - d2**-2), w2 = (d1**-2 + d2**-2)
            w1 = matrix.Minus(matrix.Mul(d1, d1).Inv(), matrix.Mul(d2, d2).Inv())
            w2 = matrix.Plus(matrix.Mul(d1, d1).Inv(), matrix.Mul(d2, d2).Inv())
			x_n = matrix.FloatVector(x.FloatArray()[:n])

			// x[n:] += mul( d1**-2 - d2**-2, x[:n])
			x_val = matrix.Mul(w1, x_n)
			x.AddIndexes(matrix.Indexes(n, x.NumElements()), x_val.FloatArray())

			// x[n:] = div( x[n:], d1**-2 + d2**-2)
			x_n = matrix.FloatVector(x.FloatArray()[n:])
			x_val = matrix.Div(x_n, w2)
			x.SetIndexes(matrix.Indexes(n, x.NumElements()), x_val.FloatArray())

			// x_n = x[:n], x-2n = x[n:2*n]
			x_n = matrix.FloatVector(x.FloatArray()[:n])
			x_2n := matrix.FloatVector(x.FloatArray()[n:2*n])

			// z := z + W^-T * G*x 
			// z[:n] += div( x[:n] - x[n:2*n], d1) 
			x_val = matrix.Div(matrix.Minus(x_n, x_2n), d1)
			z.AddIndexes(matrix.Indexes(n), x_val.FloatArray())

			// z[n:2*n] += div( -x[:n] - x[n:2*n], d2) 
			x_val = matrix.Div(matrix.Plus(x_n, x_2n).Scale(-1.0), d2)
			z.AddIndexes(matrix.Indexes(n, 2*n), x_val.FloatArray())

			// z[2*n:] += As*x[:n]
			x_val = matrix.Times(As, x_n)
			z.AddIndexes(matrix.Indexes(2*n, z.NumElements()), x_val.FloatArray())

			fmt.Printf("f end: x=\n%v\n\n", x.ToString("%.17f"))
			fmt.Printf("f end : z=\n%v\n\n", z.ToString("%.17f"))
			return nil
		}
		return
	}
	
	// matrix(n*[0.0] + n*[1.0])
	c := matrix.FloatZeros(n, 2)
	c.Set(1.0, matrix.ColumnIndexes(c, 1)...)
	matrix.Reshape(c, 2*n, 1)

	h := matrix.FloatZeros(2*n+m+1, 1)
	h.Set(1.0, 2*n)
	// h[2*n+1:] = -b
	h.SetSubMatrix(2*n+1, 0, matrix.Scale(b, -1.0))
	G := &matrixFs{A}

	//fmt.Printf("c=\n%v\n", c.ToString("%.5f"))
	//fmt.Printf("h=\n%v\n", h.ToString("%.5f"))
	
	dims := cvx.DSetNew("l", "q", "s")
	dims.Set("l", []int{2*n})
	dims.Set("q", []int{m+1})

	var solopts cvx.SolverOptions
	solopts.MaxIter = 15
	solopts.ShowProgress = true
	return cvx.ConeLpCustom(c, G, h, nil, nil, dims, Fkkt, &solopts, nil, nil)
}

func main() {
	As := "{5 5 [0.66438870630377256, 1.68511096852776343, -1.47728250254375526, 0.30317355325876538, -0.89916397951294613, 0.83465996542735588, 0.55877932252879847, -1.06626707857638992, 1.16931080498876594, -0.56601175168881845, 0.32693688563254980, -0.77989544839110070, 0.10934309320941947, -1.86725147718547602, 1.55493765723389710, -0.43138937120640264, 0.20898065620849879, -0.59006087009136965, -0.04384982450250739, 0.27861225756921282, 0.44000590962830038, 0.09061011469006654, 0.09036863350603415, 0.02113202375617339, 0.39620504458741246]}"

	bs := "{5 1 [-0.39126297858919096, -0.62890266671369610, -0.74060474150487765, -0.00313362362240900, -0.08087134031555188]}"


	var A, b *matrix.FloatMatrix = nil, nil
	flag.Parse()
	dataCount := 0
	if len(AVal) > 0 {
		dataCount += 1
		A, _ = matrix.FloatParseSpe(AVal)
		if A == nil {
			fmt.Printf("could not parse:\n%s\n", AVal)
			return
		}
	}
	if len(bVal) > 0 {
		dataCount += 1
		b, _ = matrix.FloatParseSpe(bVal)
		if b == nil {
			fmt.Printf("could not parse:\n%s\n", bVal)
			return
		}
	} 
	reftest := flag.NFlag() - dataCount > 0

	A, _ = matrix.FloatParseSpe(As)
	b, _ = matrix.FloatParseSpe(bs)

	sol, err := qcl1(A, b)
	if sol != nil {
		r := sol.Result.At("x")[0]
		x := matrix.FloatVector(r.FloatArray()[:A.Cols()])
		z := matrix.FloatVector(r.FloatArray()[A.Cols():])
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
