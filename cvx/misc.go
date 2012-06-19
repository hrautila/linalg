
package cvx

import (
	la_ "github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/linalg/blas"
	//"github.com/hrautila/go.opt/linalg/lapack"
	"github.com/hrautila/go.opt/matrix"
	"math"
)

/*
    Applies Nesterov-Todd scaling or its inverse.
    
    Computes 
    
         x := W*x        (trans is false 'N', inverse = false 'N')  
         x := W^T*x      (trans is true  'T', inverse = false 'N')  
         x := W^{-1}*x   (trans is false 'N', inverse = true  'T')  
         x := W^{-T}*x   (trans is true  'T', inverse = true  'T'). 
    
    x is a dense float matrix.
    
    W is a MatrixSet with entries:
    
    - W['dnl']: positive vector
    - W['dnli']: componentwise inverse of W['dnl']
    - W['d']: positive vector
    - W['di']: componentwise inverse of W['d']
    - W['v']: lists of 2nd order cone vectors with unit hyperbolic norms
    - W['beta']: list of positive numbers
    - W['r']: list of square matrices 
    - W['rti']: list of square matrices.  rti[k] is the inverse transpose
      of r[k].
    
    The 'dnl' and 'dnli' entries are optional, and only present when the 
    function is called from the nonlinear solver.
*/
func Scale(x *matrix.FloatMatrix, W *FloatMatrixSet, trans, inverse bool) (err error) {

	var w []*matrix.FloatMatrix
	ind := 0
	err = nil

    // Scaling for nonlinear component xk is xk := dnl .* xk; inverse 
    // scaling is xk ./ dnl = dnli .* xk, where dnl = W['dnl'], 
    // dnli = W['dnli'].

	if w = W.At("dnl"); w != nil {
		if inverse {
			w = W.At("dnli")
		}
		for k := 0; k < x.Cols(); k++ {
			err = blas.Tbmv(w[0], x, &la_.IOpt{"n", w[0].Rows()}, &la_.IOpt{"k", 0},
				&la_.IOpt{"lda", 1}, &la_.IOpt{"offsetx", k*x.Rows()})
			if err != nil { return }
		}
		ind += w[0].Rows()
	}

    // Scaling for linear 'l' component xk is xk := d .* xk; inverse 
    // scaling is xk ./ d = di .* xk, where d = W['d'], di = W['di'].

	if inverse { w = W.At("d") } else { w = W.At("di")	}
	for k := 0; k < x.Cols(); k++ {
		err = blas.Tbmv(w[0], x, &la_.IOpt{"n", w[0].Rows()}, &la_.IOpt{"k", 0},
			&la_.IOpt{"lda", 1}, &la_.IOpt{"offsetx", k*x.Rows()+ind})
		if err != nil { return }
	}
	ind += w[0].Rows()
		
    // Scaling for 'q' component is 
    //
    //    xk := beta * (2*v*v' - J) * xk
    //        = beta * (2*v*(xk'*v)' - J*xk)
    //
    // where beta = W['beta'][k], v = W['v'][k], J = [1, 0; 0, -I].
    //
    //Inverse scaling is
    //
    //    xk := 1/beta * (2*J*v*v'*J - J) * xk
    //        = 1/beta * (-J) * (2*v*((-J*xk)'*v)' + xk). 
	//wf := matrix.FloatZeros(x.Cols(), 1)
	for k, v := range W.At("v") {
		m := v.Rows()
		if inverse {
			blas.Scal(x, matrix.FScalar(-1.0),
				&la_.IOpt{"offset", ind}, &la_.IOpt{"inc", x.Rows()})
		}
		err = blas.Gemv(x, v, w[0], la_.OptTrans, &la_.IOpt{"m", m},
			&la_.IOpt{"n", x.Cols()}, &la_.IOpt{"offsetA", ind},
			&la_.IOpt{"lda", x.Rows()})
		if err != nil { return }

		err = blas.Scal(x, matrix.FScalar(-1.0),
			&la_.IOpt{"offset", ind}, &la_.IOpt{"inc", x.Rows()})
		if err != nil { return }

		err = blas.Ger(v, w[0], x, matrix.FScalar(2.0), &la_.IOpt{"m", m},
			&la_.IOpt{"n", x.Cols()}, &la_.IOpt{"lda", x.Rows()},
			&la_.IOpt{"offsetA", ind})
		if err != nil { return }

		var a float64
		if inverse {
			blas.Scal(x, matrix.FScalar(-1.0),
				&la_.IOpt{"offset", ind}, &la_.IOpt{"inc", x.Rows()})
			// a[i,j] := 1.0/W[i,j]
			a = 1.0 / W.At("beta")[0].GetIndex(k)
		} else {
			a = W.At("beta")[0].GetIndex(k)
		}
		for i := 0; i < x.Cols(); i++ {
			blas.Scal(x, matrix.FScalar(a),
				&la_.IOpt{"n", m}, &la_.IOpt{"offset", ind + i*x.Rows()})
		}
		ind += m
	}

    // Scaling for 's' component xk is
    //
    //     xk := vec( r' * mat(xk) * r )  if trans = 'N'
    //     xk := vec( r * mat(xk) * r' )  if trans = 'T'.
    //
    // r is kth element of W['r'].
    //
    // Inverse scaling is
    //
    //     xk := vec( rti * mat(xk) * rti' )  if trans = 'N'
    //     xk := vec( rti' * mat(xk) * rti )  if trans = 'T'.
    //
    // rti is kth element of W['rti'].
	maxn := 0
	for _, r := range W.At("r") {
		if r.Rows() > maxn {
			maxn = r.Rows()
		}
	}
	a := matrix.FloatZeros(maxn, maxn)
	for k, v := range W.At("r") {
		t := trans
		var r *matrix.FloatMatrix
		if inverse {
			r = v
			t = ! trans
		} else {
			r = W.At("rti")[k]
		}

		n := r.Rows()
		for i := 0; i < x.Cols(); i++ {
			// scale diagonal of xk by 0.5
			blas.Scal(x, matrix.FScalar(0.5), &la_.IOpt{"offset", ind+i*x.Rows()},
				&la_.IOpt{"inc", n+1}, &la_.IOpt{"n", n})

            // a = r*tril(x) (t is 'N') or a = tril(x)*r  (t is 'T')
			blas.Copy(r, a)
			if ! t {
				err = blas.Trmm(x, a, la_.OptRight, &la_.IOpt{"m", n},
					&la_.IOpt{"n", n}, &la_.IOpt{"lda", n}, &la_.IOpt{"ldb", n},
					&la_.IOpt{"offsetA", ind+i*x.Rows()})
				if err != nil { return }

				// x := (r*a' + a*r')  if t is 'N'
				err = blas.Syr2k(r, a, x, la_.OptNoTrans, &la_.IOpt{"n", n},
					&la_.IOpt{"k", n}, &la_.IOpt{"ldb", n}, &la_.IOpt{"ldc", n},
					&la_.IOpt{"offsetC", ind+i*x.Rows()})
				if err != nil { return }

			} else {
				err = blas.Trmm(x, a, la_.OptLeft, &la_.IOpt{"m", n},
					&la_.IOpt{"n", n}, &la_.IOpt{"lda", n}, &la_.IOpt{"ldb", n},
					&la_.IOpt{"offsetA", ind+i*x.Rows()})
				if err != nil { return }

				// x := (r'*a + a'*r)  if t is 'T'
				err = blas.Syr2k(r, a, x, la_.OptTrans, &la_.IOpt{"n", n},
					&la_.IOpt{"k", n}, &la_.IOpt{"ldb", n}, &la_.IOpt{"ldc", n},
					&la_.IOpt{"offsetC", ind+i*x.Rows()})
				if err != nil { return }
			}
		}
		ind += n*n
	}
	return 
}

// Inner product of two vectors in S.
func Sdot(x, y *matrix.FloatMatrix, dims *DimensionSet, mnl int) float64 {
	ind := mnl + dims.At("l")[0] + dims.Sum("q")
	a := blas.Dot(x, y, &la_.IOpt{"n", ind}).Float()
	for _, m := range dims.At("s") {
		aplus := blas.Dot(x, y, &la_.IOpt{"offsetx", ind}, &la_.IOpt{"offsety", ind},
			&la_.IOpt{"incx", m+1}, &la_.IOpt{"incy", m+1}, &la_.IOpt{"n", m}).Float()
		a += aplus
		for j := 0; j < m; j++ {
			aplus = 2.0 * blas.Dot(x, y, &la_.IOpt{"offsetx", ind+j}, &la_.IOpt{"offsety", ind+j},
				&la_.IOpt{"incx", m+1}, &la_.IOpt{"incy", m+1}, &la_.IOpt{"n", m-j}).Float()
			a += aplus
		}
		ind += m*m
	}
	return a
}

// Returns the norm of a vector in S
func Snrm2(x *matrix.FloatMatrix, dims *DimensionSet, mnl int) float64 {
	return math.Sqrt(Sdot(x, x, dims, mnl))
}

// Converts lower triangular matrix to symmetric.  
// Fills in the upper triangular part of the symmetric matrix stored in 
// x[offset : offset+n*n] using 'L' storage.
func Symm(x *matrix.FloatMatrix, n, offset int) (err error) {
	err = nil
	if n <= 1 {
		return
	}
	for i := 0; i < n-1; i++ {
		err = blas.Copy(x, x, &la_.IOpt{"offsetx", offset+i*(n-1)+1},
			&la_.IOpt{"offsety", offset+(i+1)*(n-1)-1}, &la_.IOpt{"incy", n},
			&la_.IOpt{"n", n-i-1})
		if err != nil { return }
	}
	return
}

func maxdim(vec []int) int {
	res := 0
	for _, v := range vec {
		if v > res {
			res = v
		}
	}
	return res
}

func maxvec(vec []float64) float64 {
	res := math.Inf(-1)
	for _, v := range vec {
		if v > res {
			res = v
		}
	}
	return res
}

func minvec(vec []float64) float64 {
	res := math.Inf(+1)
	for _, v := range vec {
		if v < res {
			res = v
		}
	}
	return res
}


// The product x := (y o x).  If diag is 'D', the 's' part of y is 
// diagonal and only the diagonal is stored.
func Sprod(x, y *matrix.FloatMatrix, dims *DimensionSet, mnl int, opts ...la_.Option) (err error){

	err = nil
	diag := la_.GetStringOpt("diag", "N", opts...)
    // For the nonlinear and 'l' blocks:  
    //
    //     yk o xk = yk .* xk.
	ind := mnl + dims.At("l")[0]
	err = blas.Tbmv(y, x, &la_.IOpt{"n", ind}, &la_.IOpt{"k", 0}, &la_.IOpt{"lda", 1})
	if err != nil { return }

    // For 'q' blocks: 
    //
    //               [ l0   l1'  ]
    //     yk o xk = [           ] * xk
    //               [ l1   l0*I ] 
    //
    // where yk = (l0, l1).
	for _, m := range dims.At("q") {
		dd := blas.Dot(x, y, &la_.IOpt{"offsetx", ind}, &la_.IOpt{"offsety", ind},
			&la_.IOpt{"n", m}).Float()
		alpha := matrix.FScalar(y.GetIndex(ind))
		blas.Scal(x, alpha, &la_.IOpt{"offset", ind+1}, &la_.IOpt{"n", m-1})
		alpha = matrix.FScalar(x.GetIndex(ind))
		blas.Axpy(x, y, alpha, &la_.IOpt{"offsetx", ind+1}, &la_.IOpt{"offsety", ind+1},
			&la_.IOpt{"n", m-1})
		x.SetIndex(ind, dd)
		ind += m
	}
	
    // For the 's' blocks:
    //
    //    yk o sk = .5 * ( Yk * mat(xk) + mat(xk) * Yk )
    // 
    // where Yk = mat(yk) if diag is 'N' and Yk = diag(yk) if diag is 'D'.

	if diag[0] == 'D' {
		maxm := maxdim(dims.At("s"))
		A := matrix.FloatZeros(maxm, maxm)
		for _, m := range dims.At("s") {
			blas.Copy(x, A, &la_.IOpt{"offsetx", ind}, &la_.IOpt{"n", m*m})
			for i := 0; i < m-1; i++ {
				err = Symm(A, m, 0)
				if err != nil { return }

				err = Symm(y, m, ind)
				if err != nil { return }
			}
			err = blas.Syr2k(A, y, x, matrix.FScalar(0.5), &la_.IOpt{"n", m}, &la_.IOpt{"k", m},
				&la_.IOpt{"lda", m}, &la_.IOpt{"ldb", m}, &la_.IOpt{"ldc", m},
				&la_.IOpt{"offsetb", ind}, &la_.IOpt{"offsetc", ind})
			if err != nil { return }
			ind += m*m
		}
	} else {
		ind2 := ind
		// !! CHECK THIS !!
		for _, m := range dims.At("s") {
			for i := 0; i < m; i++ {
				u := matrix.FloatVector(y.FloatArray()[ind2+i:ind2+m])
				u.Add(y.GetIndex(ind2+i))
				u.Mult(0.5)
				err = blas.Tbmv(u, x, &la_.IOpt{"n", m-i}, &la_.IOpt{"k", 0}, &la_.IOpt{"lda", 1},
					&la_.IOpt{"offsetx", ind+i*(m+1)})
				if err != nil { return }
			}
			ind += m*m
			ind2 += m
		}
	}
	return
}

// The product x := y o y.   The 's' components of y are diagonal and
// only the diagonals of x and y are stored.     
func Ssqr(x, y *matrix.FloatMatrix, dims *DimensionSet, mnl int) (err error) {

	blas.Copy(y, x)
	ind := mnl+dims.At("l")[0]
	err = blas.Tbmv(y, x, &la_.IOpt{"n", ind}, &la_.IOpt{"k", 0}, &la_.IOpt{"lda", 1})
	if err != nil { return }

	for _, m := range dims.At("q") {
		v := blas.Nrm2(y, &la_.IOpt{"n", m}, &la_.IOpt{"offset", ind}).Float()
		x.SetIndex(ind, v*v)
		blas.Scal(x, matrix.FScalar(2.0*y.GetIndex(ind)),
			&la_.IOpt{"n", m}, &la_.IOpt{"offset", ind})
		ind += m
	}
	err = blas.Tbmv(y, x, &la_.IOpt{"n", dims.Sum("s")}, &la_.IOpt{"k", 0}, &la_.IOpt{"lda", 1},
		&la_.IOpt{"offseta", ind}, &la_.IOpt{"offsetx", ind})
	return
}

// Returns min {t | x + t*e >= 0}, where e is defined as follows
//    
//  - For the nonlinear and 'l' blocks: e is the vector of ones.
//  - For the 'q' blocks: e is the first unit vector.
//  - For the 's' blocks: e is the identity matrix.
//    
// When called with the argument sigma, also returns the eigenvalues 
// (in sigma) and the eigenvectors (in x) of the 's' components of x.
func MaxStep(x *matrix.FloatMatrix, dims *DimensionSet, mnl int, sigma *matrix.FloatMatrix) float64 {

	t := make([]float64, 0, 10)
	ind := mnl + dims.Sum("l")
	if ind > 0 {
		t = append(t, -minvec(x.FloatArray()[:ind]))
	}
	for _, m := range dims.At("s") {
		if m > 0 {
			v := blas.Nrm2(x, &la_.IOpt{"offset", ind+1}, &la_.IOpt{"n", m-1}).Float()
			v -= x.GetIndex(ind)
			t = append(t, v)
		}
		ind += m
	}
	var Q *matrix.FloatMatrix
	var w *matrix.FloatMatrix
	ind2 := 0
	for _, m := range dims.At("s") {
		if sigma == nil {
			blas.Copy(x, Q, &la_.IOpt{"offsetx", ind}, &la_.IOpt{"n", m*m})
			// !! CHECK THIS !!
			//lapack.Syevr(Q, w, nil, []int{1,1}, &la_.OptRangeInt, &la_.IOpt{"n", m},
			//	&la_.IOpt{"lda", m})
			if m > 0 {
				t = append(t, -w.GetIndex(0))
			}
		} else {
			// !! CHECK THIS !!
			//lapack.Syevr(Q, sigma, nil, nil, &la_.OptJobzV, &la_.IOpt{"n", m},
			//	&la_.IOpt{"lda", m}, &la_.IOpt{"offseta", ind}, &la_.IOpt{"offsetw", ind2})
			if m > 0 {
				t = append(t, -sigma.GetIndex(ind2))
			}
		}
		ind += m*m
		ind2 += m
	}
	if len(t) > 0 {
		return maxvec(t)
	}
	return 0.0
}

/*
     Copy x to y using packed storage.
    
     The vector x is an element of S, with the 's' components stored in 
     unpacked storage.  On return, x is copied to y with the 's' components
     stored in packed storage and the off-diagonal entries scaled by 
     sqrt(2).
 */
func Pack(x, y *matrix.FloatMatrix, dims *DimensionSet, opts ...la_.Option) (err error) {
	err = nil
	mnl := la_.GetIntOpt("mnl", 0, opts...)
	offsetx := la_.GetIntOpt("offsetx", 0, opts...)
	offsety := la_.GetIntOpt("offsety", 0, opts...)

	nlq := mnl + dims.At("l")[0] + dims.Sum("q")
	blas.Copy(x, y, &la_.IOpt{"n", nlq}, &la_.IOpt{"offsetx", offsetx},
		&la_.IOpt{"offsety", offsety})
	iu, ip := offsetx + nlq, offsety + nlq
	for _, n := range dims.At("s") {
		for k := 0; k < n; k++ {
			blas.Copy(x, y, &la_.IOpt{"n", n-k}, &la_.IOpt{"offsetx", iu+k*(n+1)},
				&la_.IOpt{"offsety", ip})
			y.SetIndex(ip, (y.GetIndex(ip) / math.Sqrt(2.0)))
		}
		iu += n*n
	}
	np := dims.SumPacked("s")
	blas.Scal(y, matrix.FScalar(math.Sqrt(2.0)), &la_.IOpt{"n", np}, &la_.IOpt{"offset", offsety+nlq})
	return
}

/*
     The vector x is an element of S, with the 's' components stored
     in unpacked storage and off-diagonal entries scaled by sqrt(2).
     On return, x is copied to y with the 's' components stored in 
     unpacked storage.

 */
func UnPack(x, y *matrix.FloatMatrix, dims *DimensionSet, opts ...la_.Option) (err error) {
	err = nil
	mnl := la_.GetIntOpt("mnl", 0, opts...)
	offsetx := la_.GetIntOpt("offsetx", 0, opts...)
	offsety := la_.GetIntOpt("offsety", 0, opts...)

	nlq := mnl + dims.At("l")[0] + dims.Sum("q")
	err = blas.Copy(x, y, &la_.IOpt{"n", nlq}, &la_.IOpt{"offsetx", offsetx},
		&la_.IOpt{"offsety", offsety})
	if err != nil { return }

	iu, ip := offsetx + nlq, offsety + nlq
	for _, n := range dims.At("s") {
		for k := 0; k < n; k++ {
			err = blas.Copy(x, y, &la_.IOpt{"n", n-k}, &la_.IOpt{"offsetx", ip},
				&la_.IOpt{"offsety",  iu+k*(n+1)})
			if err != nil { return }

			y.SetIndex(ip, (y.GetIndex(ip) * math.Sqrt(2.0)))
		}
		iu += n*n
	}
	nu := dims.SumSquared("s")
	err = blas.Scal(y, matrix.FScalar(1.0/math.Sqrt(2.0)),
		&la_.IOpt{"n", nu}, &la_.IOpt{"offset", offsety+nlq})
	return
}

/*
    Returns the Nesterov-Todd scaling W at points s and z, and stores the 
    scaled variable in lmbda. 
    
        W * z = W^{-T} * s = lmbda. 
 */
func ComputeScaling(s, z, lambda *matrix.FloatMatrix, dims *DimensionSet, mnl int) (W *FloatMatrixSet, err error) {
	err = nil
	W = FloatSetNew("dnl", "dnli", "d", "di", "v", "beta", "r", "rti")
	return 
}

// Local Variables:
// tab-width: 4
// End:
