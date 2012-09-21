
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/cvx package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package cvx

import (
	la "github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/cvx/sets"
	"errors"
	"fmt"
	"math"
)

// Implements MatrixP interface for standard matrix valued P.
type matrixP struct {
	mP *matrix.FloatMatrix
}

func (m *matrixP) Pf(x, y *matrix.FloatMatrix, alpha, beta float64) error {
	return blas.SymvFloat(m.mP, x, y, alpha, beta)
}


func checkConeQpDimensions(dims *sets.DimensionSet) error {
	if len(dims.At("l")) < 1 {
		dims.Set("l", []int{0})
	} else	if dims.At("l")[0] < 0 {
		return errors.New("dimension 'l' must be nonnegative integer")
	}
	for _, m := range dims.At("q") {
		if m < 1 {
			return errors.New("dimension 'q' must be list of positive integers")
		}
	}
	for _, m := range dims.At("s") {
		if m < 0 {
			return errors.New("dimension 's' must be list of nonnegative integers")
		}
	}
	return nil
}

// Solves a pair of primal and dual convex quadratic cone programs
//
//        minimize    (1/2)*x'*P*x + q'*x    
//        subject to  G*x + s = h      
//                    A*x = b
//                    s >= 0
//
//        maximize    -(1/2)*(q + G'*z + A'*y)' * pinv(P) * (q + G'*z + A'*y)
//                    - h'*z - b'*y 
//        subject to  q + G'*z + A'*y in range(P)
//                    z >= 0.
//
// The inequalities are with respect to a cone C defined as the Cartesian
// product of N + M + 1 cones:
//    
//        C = C_0 x C_1 x .... x C_N x C_{N+1} x ... x C_{N+M}.
//
// The first cone C_0 is the nonnegative orthant of dimension ml.  
// The next N cones are 2nd order cones of dimension mq[0], ..., mq[N-1].
// The second order cone of dimension m is defined as
//    
//        { (u0, u1) in R x R^{m-1} | u0 >= ||u1||_2 }.
//
// The next M cones are positive semidefinite cones of order ms[0], ...,
// ms[M-1] >= 0.  
//
func ConeQp(P, q, G, h, A, b *matrix.FloatMatrix, dims *sets.DimensionSet, solopts *SolverOptions, initvals *sets.FloatMatrixSet) (sol *Solution, err error) {

	if q == nil || q.Cols() != 1 {
		err = errors.New("'q' must be non-nil matrix with one column")
		return
	}
	if P == nil || P.Rows() != q.Rows() || P.Cols() != q.Rows() {
		err = errors.New(fmt.Sprintf("'P' must be non-nil matrix of size (%d, %d)",
			q.Rows(), q.Rows()))
		return
	}

	if h == nil {
		h = matrix.FloatZeros(0, 1)
	}
	if h.Cols() != 1 {
		err = errors.New("'h' must be non-nil matrix with one column")
		return
	}
	if dims == nil {
		dims = sets.NewDimensionSet("l", "q", "s")
		dims.Set("l", []int{h.Rows()})
	}

	err = checkConeQpDimensions(dims)
	if err != nil {
		return
	}

	cdim := dims.Sum("l", "q") + dims.SumSquared("s")
	//cdim_pckd := dims.Sum("l", "q") + dims.SumPacked("s")
	//cdim_diag := dims.Sum("l", "q", "s")

	if h.Rows() != cdim {
		err = errors.New(fmt.Sprintf("'h' must be float matrix of size (%d,1)", cdim))
		return 
	}

	if G == nil {
		G = matrix.FloatZeros(0, q.Rows())
	}
	if !G.SizeMatch(cdim, q.Rows()) {
		estr := fmt.Sprintf("'G' must be of size (%d,%d)", cdim, q.Rows())
		err = errors.New(estr)
		return 
	}

	// Check A and set defaults if it is nil
	if A == nil {
		// zeros rows reduces Gemv to vector products
		A = matrix.FloatZeros(0, q.Rows())
	}
	if A.Cols() != q.Rows() {
		estr := fmt.Sprintf("'A' must have %d columns", q.Rows())
		err = errors.New(estr)
		return 
	}

	// Check b and set defaults if it is nil
	if b == nil {
		b = matrix.FloatZeros(0, 1)
	}
	if b.Cols() != 1 {
		estr := fmt.Sprintf("'b' must be a matrix with 1 column")
		err = errors.New(estr)
		return 
	}
	if b.Rows() != A.Rows() {
		estr := fmt.Sprintf("'b' must have length %d", A.Rows())
		err = errors.New(estr)
		return 
	}

	var mA = matrixA{A}
	var mG = matrixG{G, dims}
	var mP = matrixP{P}
	
	solvername := solopts.KKTSolverName
	if len(solvername) == 0 {
		if len(dims.At("q")) > 0 || len(dims.At("s")) > 0 {
			solvername = "qr"
		} else {
			solvername = "chol2"
		}
	}

	var factor kktFactor
	var kktsolver CustomKKT = nil
	if kktfunc, ok := solvers[solvername]; ok {
		// kkt function returns us problem spesific factor function.
		factor, err = kktfunc(G, dims, A, 0)
		kktsolver = func(W *sets.FloatMatrixSet) (KKTFunc, error) {
			return factor(W, P, nil)
		}
	} else {
		err = errors.New(fmt.Sprintf("solver '%s' not known", solvername))
		return
	}
	return ConeQpCustom(&mP, q, &mG, h, &mA, b, dims, kktsolver, solopts, initvals)
}


// Solves a pair of primal and dual convex quadratic cone programs using custom KKT solver.
//
// The customized solver provides a routine for solving linear  equations (`KKT systems')
//
//            [ P   A'  G'    ] [ ux ]   [ bx ]
//            [ A   0   0     ] [ uy ] = [ by ].
//            [ G   0   -W'*W ] [ uz ]   [ bz ]
//
// W is a scaling matrix, a block diagonal mapping
//
//           W*u = ( W0*u_0, ..., W_{N+M}*u_{N+M} )
//
// defined as follows.
//
// ** For the 'l' block (W_0):
//
//              W_0 = diag(d),
//
// with d a positive vector of length ml.
//
// ** For the 'q' blocks (W_{k+1}, k = 0, ..., N-1):
//
//              W_{k+1} = beta_k * ( 2 * v_k * v_k' - J )
//
// where beta_k is a positive scalar, v_k is a vector in R^mq[k]
// with v_k[0] > 0 and v_k'*J*v_k = 1, and J = [1, 0; 0, -I].
//
// ** For the 's' blocks (W_{k+N}, k = 0, ..., M-1):
//
//              W_k * u = vec(r_k' * mat(u) * r_k)
//
// where r_k is a nonsingular matrix of order ms[k], and mat(x) is
// the inverse of the vec operation.
//
// The optional argument kktsolver is a function that will be
// called as g = kktsolver(W).  W is a FloatMatrixSet that contains
// the parameters of the scaling:
//
// - W['d'] is a positive 'd' matrix of size (ml,1).
// - W['di'] is a positive 'd' matrix with the elementwise inverse of W['d'].
// - W['beta'] is a matrix [ beta_0, ..., beta_{N-1} ]
// - W['v'] is a list [ v_0, ..., v_{N-1} ] of float matrices.
// - W['r'] is a list [ r_0, ..., r_{M-1} ] of matrices
// - W['rti'] is a list [ rti_0, ..., rti_{M-1} ], with rti_k the inverse of the
//   transpose of r_k.
//
// The call g = kktsolver(W) should return a function g that solves 
// the KKT system by g(x, y, z).  On entry, x, y, z contain the 
// righthand side bx, by, bz.  On exit, they contain the solution,
// with uz scaled, the argument z contains W*uz.  In other words, 
// on exit x, y, z are the solution of
//
//            [ P   A'  G'*W^{-1} ] [ ux ]   [ bx ]
//            [ A   0   0         ] [ uy ] = [ by ].
//            [ G   0   -W'       ] [ uz ]   [ bz ]
//
//
func ConeQpKKT(P, q, G, h, A, b *matrix.FloatMatrix, dims *sets.DimensionSet, kktsolver CustomKKT, solopts *SolverOptions, initvals *sets.FloatMatrixSet) (sol *Solution, err error) {

	if q == nil || q.Cols() != 1 {
		err = errors.New("'q' must be non-nil matrix with one column")
		return
	}
	if P == nil || P.Rows() != q.Rows() || P.Cols() != q.Rows() {
		err = errors.New(fmt.Sprintf("'P' must be non-nil matrix of size (%d, %d)",
			q.Rows(), q.Rows()))
		return
	}

	if h == nil {
		h = matrix.FloatZeros(0, 1)
	}
	if h.Cols() != 1 {
		err = errors.New("'h' must be non-nil matrix with one column")
		return
	}
	if dims == nil {
		dims = sets.NewDimensionSet("l", "q", "s")
		dims.Set("l", []int{h.Rows()})
	}

	err = checkConeQpDimensions(dims)
	if err != nil {
		return
	}

	cdim := dims.Sum("l", "q") + dims.SumSquared("s")
	//cdim_pckd := dims.Sum("l", "q") + dims.SumPacked("s")
	//cdim_diag := dims.Sum("l", "q", "s")

	if h.Rows() != cdim {
		err = errors.New(fmt.Sprintf("'h' must be float matrix of size (%d,1)", cdim))
		return 
	}

	if G == nil {
		G = matrix.FloatZeros(0, q.Rows())
	}
	if !G.SizeMatch(cdim, q.Rows()) {
		estr := fmt.Sprintf("'G' must be of size (%d,%d)", cdim, q.Rows())
		err = errors.New(estr)
		return 
	}

	// Check A and set defaults if it is nil
	if A == nil {
		// zeros rows reduces Gemv to vector products
		A = matrix.FloatZeros(0, q.Rows())
	}
	if A.Cols() != q.Rows() {
		estr := fmt.Sprintf("'A' must have %d columns", q.Rows())
		err = errors.New(estr)
		return 
	}

	// Check b and set defaults if it is nil
	if b == nil {
		b = matrix.FloatZeros(0, 1)
	}
	if b.Cols() != 1 {
		estr := fmt.Sprintf("'b' must be a matrix with 1 column")
		err = errors.New(estr)
		return 
	}
	if b.Rows() != A.Rows() {
		estr := fmt.Sprintf("'b' must have length %d", A.Rows())
		err = errors.New(estr)
		return 
	}

	var mA = matrixA{A}
	var mG = matrixG{G, dims}
	var mP = matrixP{P}
	
	if kktsolver == nil {
		err = errors.New("nil kktsolver not allowed")
		return
	}

	return ConeQpCustom(&mP, q, &mG, h, &mA, b, dims, kktsolver, solopts, initvals)
}

// Solves a pair of primal and dual cone programs using custom KKT solver and custom
// matrices P, G and A.
//
// P must implement interface MatrixP, G must implement interface MatrixG
// and A must implement interface MatrixA.
//
func ConeQpCustom(P MatrixP, q *matrix.FloatMatrix, G MatrixG, h *matrix.FloatMatrix,
	A MatrixA, b *matrix.FloatMatrix, dims *sets.DimensionSet, kktsolver CustomKKT,
	solopts *SolverOptions, initvals *sets.FloatMatrixSet) (sol *Solution, err error) {


	err = nil
	EXPON := 3
	STEP := 0.99

	sol = &Solution{Unknown,
		/*nil, nil, nil, nil,*/ nil,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0}

	//var kktsolver func(*sets.FloatMatrixSet)(KKTFunc, error) = nil
	var refinement int
	var correction bool = true

	feasTolerance := FEASTOL
	absTolerance := ABSTOL
	relTolerance := RELTOL
	if solopts.FeasTol > 0.0 {
		feasTolerance = solopts.FeasTol
	}
	if solopts.AbsTol > 0.0 {
		absTolerance = solopts.AbsTol
	}
	if solopts.RelTol > 0.0 {
		relTolerance = solopts.RelTol
	}

	if q == nil || q.Cols() != 1 {
		err = errors.New("'q' must be non-nil matrix with one column")
		return
	}

	if h == nil {
		h = matrix.FloatZeros(0, 1)
	}
	if h.Cols() != 1 {
		err = errors.New("'h' must be non-nil matrix with one column")
		return
	}
	if dims == nil {
		dims = sets.NewDimensionSet("l", "q", "s")
		dims.Set("l", []int{h.Rows()})
	}

	err = checkConeQpDimensions(dims)
	if err != nil {
		return
	}

	cdim := dims.Sum("l", "q") + dims.SumSquared("s")
	//cdim_pckd := dims.Sum("l", "q") + dims.SumPacked("s")
	cdim_diag := dims.Sum("l", "q", "s")

	if h.Rows() != cdim {
		err = errors.New(fmt.Sprintf("'h' must be float matrix of size (%d,1)", cdim))
		return 
	}

	// Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G.
	indq := make([]int, 0)
	indq = append(indq, dims.At("l")[0])
	for _, k := range dims.At("q") {
		indq = append(indq, indq[len(indq)-1]+k)
	}

    // Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G.
	inds := make([]int, 0)
	inds = append(inds, indq[len(indq)-1])
	for _, k := range dims.At("s") {
		inds = append(inds, inds[len(inds)-1]+k*k)
	}

	if P == nil {
		err = errors.New("'P' must be non-nil MatrixP interface.")
		return
	}
	fP := func(u, v *matrix.FloatMatrix, alpha, beta float64) error{
		return P.Pf(u, v, alpha, beta)
	}

	if G == nil {
		err = errors.New("'G' must be non-nil MatrixG interface.")
		return
	}
	fG := func(x, y *matrix.FloatMatrix, alpha, beta float64, trans la.Option) error{
		return G.Gf(x, y, alpha, beta, trans)
	}

	var fA func(x, y *matrix.FloatMatrix, alpha, beta float64, trans la.Option) error = nil
	var Adummy *matrix.FloatMatrix

	// Check A and set defaults if it is nil
	if A == nil {
		// zeros rows reduces Gemv to vector products
		Adummy = matrix.FloatZeros(0, q.Rows())
		fA = func(x, y *matrix.FloatMatrix, alpha, beta float64, trans la.Option) error {
			return blas.GemvFloat(Adummy, x, y, alpha, beta, trans)
		}
	} else {
		fA = func(x, y *matrix.FloatMatrix, alpha, beta float64, trans la.Option) error {
			return A.Af(x, y, alpha, beta, trans)
		}
	}

	// Check b and set defaults if it is nil
	if b == nil {
		b = matrix.FloatZeros(0, 1)
	}
	if b.Cols() != 1 {
		estr := fmt.Sprintf("'b' must be a matrix with 1 column")
		err = errors.New(estr)
		return 
	}

	if b.Rows() > q.Rows() {
		err = errors.New("1: Rank(A) < p or Rank[G; A] < n")
		return 
	}

    // kktsolver(W) returns a routine for solving 3x3 block KKT system 
    //
    //     [ 0   A'  G'*W^{-1} ] [ ux ]   [ bx ]
    //     [ A   0   0         ] [ uy ] = [ by ].
    //     [ G   0   -W'       ] [ uz ]   [ bz ]

	if kktsolver == nil {
		err = errors.New("nil kktsolver not allowed.")
		return
	}

	ws3 := matrix.FloatZeros(cdim, 1)
	wz3 := matrix.FloatZeros(cdim, 1)

	// 
	res := func(ux, uy, uz, us, vx, vy, vz, vs *matrix.FloatMatrix, W *sets.FloatMatrixSet, lmbda *matrix.FloatMatrix) (err error) {
        // Evaluates residual in Newton equations:
        // 
        //      [ vx ]    [ vx ]   [ 0     ]   [ P  A'  G' ]   [ ux        ]
        //      [ vy ] := [ vy ] - [ 0     ] - [ A  0   0  ] * [ uy        ]
        //      [ vz ]    [ vz ]   [ W'*us ]   [ G  0   0  ]   [ W^{-1}*uz ]
        //
        //      vs := vs - lmbda o (uz + us).

        // vx := vx - P*ux - A'*uy - G'*W^{-1}*uz
		fP(ux, vx, -1.0, 1.0)
		fA(uy, vx, -1.0, 1.0, la.OptTrans)
		blas.Copy(uz, wz3)
		scale(wz3, W, true, false)
		fG(wz3, vx, -1.0, 1.0, la.OptTrans)
        // vy := vy - A*ux
        fA(ux, vy, -1.0, 1.0, la.OptNoTrans)

        // vz := vz - G*ux - W'*us
        fG(ux, vz, -1.0, 1.0, la.OptNoTrans)
        blas.Copy(us, ws3)
        scale(ws3, W, true, false)
        blas.AxpyFloat(ws3, vz, -1.0)
 
        // vs := vs - lmbda o (uz + us)
        blas.Copy(us, ws3)
        blas.AxpyFloat(uz, ws3, 1.0)
        sprod(ws3, lmbda, dims, 0, la.OptDiag)
        blas.AxpyFloat(ws3, vs, -1.0)
		return 
	}

	resx0 := math.Max(1.0, math.Sqrt(blas.Dot(q,q).Float()))
	resy0 := math.Max(1.0, math.Sqrt(blas.Dot(b,b).Float()))
	resz0 := math.Max(1.0, snrm2(h, dims, 0))
	//fmt.Printf("resx0: %.17f, resy0: %.17f, resz0: %.17f\n", resx0, resy0, resz0)

	var x, y, z, s, dx, dy, ds, dz, rx, ry, rz *matrix.FloatMatrix
	var lmbda, lmbdasq, sigs, sigz *matrix.FloatMatrix
	var W *sets.FloatMatrixSet
	var f, f3 KKTFunc
	var resx, resy, resz, step, sigma, mu, eta float64
	var gap, pcost, dcost, relgap, pres, dres, f0 float64

	if cdim == 0 {
		// Solve
		//
		//     [ P  A' ] [ x ]   [ -q ]
		//     [       ] [   ] = [    ].
		//     [ A  0  ] [ y ]   [  b ]
		//
		Wtmp := sets.NewFloatSet("d", "di", "beta", "v", "r", "rti")
		Wtmp.Set("d", matrix.FloatZeros(0, 1))
		Wtmp.Set("di", matrix.FloatZeros(0, 1))
		f3, err = kktsolver(Wtmp)
		if err != nil {
			s := fmt.Sprintf("kkt error: %s", err)
			err = errors.New("2: Rank(A) < p or Rank(([P; A; G;]) < n : "+s)
			return
		}
		x = q.Copy()
		blas.ScalFloat(x, 0.0)
		y = b.Copy()
		f3(x, y, matrix.FloatZeros(0, 1))
		
		// dres = || P*x + q + A'*y || / resx0 
		rx = q.Copy()
		fP(x, rx, 1.0, 1.0)
		pcost = 0.5 *( blas.DotFloat(x, rx) + blas.DotFloat(x, q))
		fA(y, rx, 1.0, 1.0, la.OptTrans)
		dres = math.Sqrt(blas.DotFloat(rx, rx)/resx0)
		
		ry = b.Copy()
		fA(x, ry, 1.0, -1.0, la.OptNoTrans)
		pres = math.Sqrt(blas.DotFloat(ry, ry)/resy0)

		relgap = 0.0
		if pcost == 0.0 {
			relgap = math.NaN()
		}

		sol.Result = sets.NewFloatSet("x", "y", "s", "z")
		sol.Result.Set("x", x)
		sol.Result.Set("y", y)
		sol.Result.Set("s", matrix.FloatZeros(0,1))
		sol.Result.Set("z", matrix.FloatZeros(0,1))
		sol.Status = Optimal
		sol.Gap = 0.0; sol.RelativeGap = relgap
		sol.PrimalObjective = pcost
		sol.DualObjective = pcost
		sol.PrimalInfeasibility = pres
		sol.DualInfeasibility = dres
		sol.PrimalSlack = 0.0
		sol.DualSlack = 0.0
		return
	}
	x = q.Copy()
	y = b.Copy()
	s = matrix.FloatZeros(cdim, 1)
	z = matrix.FloatZeros(cdim, 1)

	var ts, tz, nrms, nrmz float64

	if initvals == nil {
		// Factor
		//
		//     [ 0   A'  G' ] 
		//     [ A   0   0  ].
		//     [ G   0  -I  ]
		//
		W = sets.NewFloatSet("d", "di", "v", "beta", "r", "rti")
		W.Set("d", matrix.FloatOnes(dims.At("l")[0], 1))
		W.Set("di", matrix.FloatOnes(dims.At("l")[0], 1))
		W.Set("beta", matrix.FloatOnes(len(dims.At("q")), 1))

		for _, n := range dims.At("q")  {
			vm := matrix.FloatZeros(n, 1)
			vm.SetIndex(0, 1.0)
			W.Append("v", vm)
		}
		for _, n := range dims.At("s") {
			W.Append("r", matrix.FloatIdentity(n))
			W.Append("rti", matrix.FloatIdentity(n))
		}
		f, err = kktsolver(W)
		if err != nil {
			s := fmt.Sprintf("kkt error: %s", err)
			err = errors.New("3: Rank(A) < p or Rank([P; G; A]) < n : "+s)
			return 
		}
		// Solve
		//
		//     [ P   A'  G' ]   [ x ]   [ -q ]
		//     [ A   0   0  ] * [ y ] = [  b ].
		//     [ G   0  -I  ]   [ z ]   [  h ]
		x = q.Copy()
		blas.ScalFloat(x, -1.0)
		y = b.Copy()
		z = h.Copy()
		err = f(x, y, z)
		if err != nil {
			s := fmt.Sprintf("kkt error: %s", err)
			err = errors.New("4: Rank(A) < p or Rank([P; G; A]) < n : "+s)
			return 
		}
		s = z.Copy()
		blas.ScalFloat(s, -1.0)

		nrms = snrm2(s, dims, 0)
		ts,_ = maxStep(s, dims, 0, nil)
		if ts >= -1e-8 * math.Max(nrms, 1.0) {
			// a = 1.0 + ts  
			a := 1.0 + ts
			is := make([]int, 0)
			// indexes s[:dims['l']]
			is = append(is, matrix.MakeIndexSet(0, dims.At("l")[0], 1)...)
			// indexes s[indq[:-1]]
			is = append(is, indq[:len(indq)-1]...)
			ind := dims.Sum("l", "q")
			// indexes s[ind:ind+m*m:m+1] (diagonal)
			for _, m := range dims.At("s") {
				is = append(is, matrix.MakeIndexSet(ind, ind+m*m, m+1)...)
				ind += m*m
			}
			for _, k := range is {
				s.SetIndex(k, a + s.GetIndex(k))
			}
		}

		nrmz = snrm2(z, dims, 0)
		tz,_ = maxStep(z, dims, 0, nil)
		if tz >= -1e-8 * math.Max(nrmz, 1.0) {
			a := 1.0 + tz
			is := make([]int, 0)
			is = append(is, matrix.MakeIndexSet(0, dims.At("l")[0], 1)...)
			is = append(is, indq[:len(indq)-1]...)
			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				is = append(is, matrix.MakeIndexSet(ind, ind+m*m, m+1)...)
				ind += m*m
			}
			for _, k := range is {
				z.SetIndex(k, a + z.GetIndex(k))
			}
		}

	} else {
		ix := initvals.At("x")[0]
		if ix != nil {
			blas.Copy(ix, x)
		} else {
			blas.ScalFloat(x, 0.0)
		}

		is := initvals.At("s")[0]
		if is != nil {
			blas.Copy(is, s)
		} else {
			iset := make([]int, 0)
			iset = append(iset, matrix.MakeIndexSet(0, dims.At("l")[0], 1)...)
			iset = append(iset, indq[:len(indq)-1]...)
			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				iset = append(iset, matrix.MakeIndexSet(ind, ind+m*m, m+1)...)
				ind += m*m
			}
			for _, k := range iset {
				s.SetIndex(k, 1.0)
			}
		}
		
		iy := initvals.At("y")[0]
		if iy != nil {
			blas.Copy(iy, y)
		} else {
			blas.ScalFloat(y, 0.0)
		}

		iz := initvals.At("z")[0]
		if iz != nil {
			blas.Copy(iz, z)
		} else {
			iset := make([]int, 0)
			iset = append(iset, matrix.MakeIndexSet(0, dims.At("l")[0], 1)...)
			iset = append(iset, indq[:len(indq)-1]...)
			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				iset = append(iset, matrix.MakeIndexSet(ind, ind+m*m, m+1)...)
				ind += m*m
			}
			for _, k := range iset {
				z.SetIndex(k, 1.0)
			}
		}
	}

	rx = q.Copy()
	ry = b.Copy()
	rz = matrix.FloatZeros(cdim, 1)
	dx = x.Copy()
	dy = y.Copy()
	dz = matrix.FloatZeros(cdim, 1)
	ds = matrix.FloatZeros(cdim, 1)
	lmbda = matrix.FloatZeros(cdim_diag, 1)
	lmbdasq = matrix.FloatZeros(cdim_diag, 1)
	sigs = matrix.FloatZeros(dims.Sum("s"), 1)
	sigz = matrix.FloatZeros(dims.Sum("s"), 1)

	var WS fClosure

	gap = sdot(s, z, dims, 0)
	for iter := 0; iter < solopts.MaxIter+1; iter++ {

        // f0 = (1/2)*x'*P*x + q'*x + r and  rx = P*x + q + A'*y + G'*z.
        blas.Copy(q, rx)
        fP(x, rx, 1.0, 1.0)
        f0 = 0.5 * (blas.DotFloat(x, rx) + blas.DotFloat(x, q))
        fA(y, rx, 1.0, 1.0, la.OptTrans)
        fG(z, rx, 1.0, 1.0, la.OptTrans)
        resx = math.Sqrt(blas.DotFloat(rx, rx))
           
        // ry = A*x - b
        blas.Copy(b, ry)
        fA(x, ry, 1.0, -1.0, la.OptNoTrans)
        resy = math.Sqrt(blas.DotFloat(ry, ry))

        // rz = s + G*x - h
        blas.Copy(s, rz)
        blas.AxpyFloat(h, rz, -1.0)
        fG(x, rz, 1.0, 1.0, la.OptNoTrans)
        resz = snrm2(rz, dims, 0)
		//fmt.Printf("resx: %.17f, resy: %.17f, resz: %.17f\n", resx, resy, resz)

        // Statistics for stopping criteria.

        // pcost = (1/2)*x'*P*x + q'*x 
        // dcost = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h) '
        //       = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h+s) - z'*s
        //       = (1/2)*x'*P*x + q'*x + y'*ry + z'*rz - gap
        pcost = f0
        dcost = f0 + blas.DotFloat(y, ry) + sdot(z, rz, dims, 0) - gap
        if pcost < 0.0 {
            relgap = gap / -pcost
        } else if dcost > 0.0 {
            relgap = gap / dcost 
        } else {
            relgap = math.NaN()
		}
        pres = math.Max(resy/resy0, resz/resz0)
        dres = resx/resx0 

		if solopts.ShowProgress {
			if iter == 0 {
				// show headers of something 
				fmt.Printf("% 10s% 12s% 10s% 8s% 7s\n",
					"pcost", "dcost", "gap", "pres", "dres")
			}
			// show something
            fmt.Printf("%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e\n",
				iter, pcost, dcost, gap, pres, dres)
		}

		
		if pres <= feasTolerance && dres <= feasTolerance &&
			( gap <= absTolerance || (!math.IsNaN(relgap) && relgap <= relTolerance)) ||
			iter == solopts.MaxIter {

			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				symm(s, m, ind)
				symm(z, m, ind)
				ind += m*m
			}
			ts,_ = maxStep(s, dims, 0, nil)
			tz,_ = maxStep(z, dims, 0, nil)
			if iter == solopts.MaxIter {
				// terminated on max iterations.
				sol.Status = Unknown
				err = errors.New("Terminated (maximum iterations reached)")
				fmt.Printf("Terminated (maximum iterations reached)\n")
				return
			}
			// optimal solution found
			//fmt.Print("Optimal solution.\n")
			err = nil
			sol.Result = sets.NewFloatSet("x", "y", "s", "z")
			sol.Result.Set("x", x)
			sol.Result.Set("y", y)
			sol.Result.Set("s", s)
			sol.Result.Set("z", z)
			sol.Status = Optimal
			sol.Gap = gap; sol.RelativeGap = relgap
			sol.PrimalObjective = pcost
			sol.DualObjective = dcost
			sol.PrimalInfeasibility = pres
			sol.DualInfeasibility = dres
			sol.PrimalSlack = -ts
			sol.DualSlack = -tz
			sol.PrimalResidualCert = math.NaN()
			sol.DualResidualCert = math.NaN()
			sol.Iterations = iter
			return
		}

        // Compute initial scaling W and scaled iterates:  
        //
        //     W * z = W^{-T} * s = lambda.
        // 
        // lmbdasq = lambda o lambda.
		if iter == 0 {
			W, err = computeScaling(s, z, lmbda, dims, 0)
		}
		ssqr(lmbdasq, lmbda, dims, 0)

		f3, err = kktsolver(W)
		if err != nil {
			if iter == 0 {
				s := fmt.Sprintf("kkt error: %s", err)
				err = errors.New("5: Rank(A) < p or Rank([P; A; G]) < n : "+s)
				return
			} else {
				ind := dims.Sum("l", "q")
				for _, m := range dims.At("s") {
					symm(s, m, ind)
					symm(z, m, ind)
					ind += m*m
				}
				ts,_ = maxStep(s, dims, 0, nil)
				tz,_ = maxStep(z, dims, 0, nil)
				// terminated (singular KKT matrix)
				fmt.Printf("Terminated (singular KKT matrix).\n")
				err = errors.New("Terminated (singular KKT matrix).")
				sol.Result = sets.NewFloatSet("x", "y", "s", "z")
				sol.Result.Set("x", x)
				sol.Result.Set("y", y)
				sol.Result.Set("s", s)
				sol.Result.Set("z", z)
				sol.Status = Unknown
				sol.RelativeGap = relgap
				sol.PrimalObjective = pcost
				sol.DualObjective = dcost
				sol.PrimalInfeasibility = pres
				sol.DualInfeasibility = dres
				sol.PrimalSlack = -ts
				sol.DualSlack = -tz
				sol.Iterations = iter
				return
			}
		}
		// f4_no_ir(x, y, z, s) solves
        // 
        //     [ 0     ]   [ P  A'  G' ]   [ ux        ]   [ bx ]
        //     [ 0     ] + [ A  0   0  ] * [ uy        ] = [ by ]
        //     [ W'*us ]   [ G  0   0  ]   [ W^{-1}*uz ]   [ bz ]
        //
        //     lmbda o (uz + us) = bs.
        //
        // On entry, x, y, z, s contain bx, by, bz, bs.
        // On exit, they contain ux, uy, uz, us.

		f4_no_ir := func(x, y, z, s *matrix.FloatMatrix)error {
            // Solve 
            //
            //     [ P A' G'   ] [ ux        ]    [ bx                    ]
            //     [ A 0  0    ] [ uy        ] =  [ by                    ]
            //     [ G 0 -W'*W ] [ W^{-1}*uz ]    [ bz - W'*(lmbda o\ bs) ]
            //
            //     us = lmbda o\ bs - uz.
            //
            // On entry, x, y, z, s  contains bx, by, bz, bs. 
            // On exit they contain x, y, z, s.
            
            // s := lmbda o\ s 
            //    = lmbda o\ bs
			sinv(s, lmbda, dims, 0)

            // z := z - W'*s 
            //    = bz - W'*(lambda o\ bs)
			blas.Copy(s, ws3)
			scale(ws3, W, true, false)
			blas.AxpyFloat(ws3, z, -1.0)

			err := f3(x, y, z)
			if err != nil {
				return err
			}

            // s := s - z 
            //    = lambda o\ bs - uz.
			blas.AxpyFloat(z, s, -1.0)
			return nil
		}

		if iter == 0 {
			if refinement > 0 || solopts.Debug {
				WS.wx = q.Copy()
				WS.wy = y.Copy()
				WS.ws = matrix.FloatZeros(cdim, 1)
				WS.wz = matrix.FloatZeros(cdim, 1)
			}
			if refinement > 0 {
				WS.wx2 = q.Copy()
				WS.wy2 = y.Copy()
				WS.ws2 = matrix.FloatZeros(cdim, 1)
				WS.wz2 = matrix.FloatZeros(cdim, 1)
			}
		}

		f4 := func(x, y, z, s *matrix.FloatMatrix)(err error) {
			err = nil
			if refinement > 0 || solopts.Debug {
				blas.Copy(x, WS.wx)
				blas.Copy(y, WS.wy)
				blas.Copy(z, WS.wz)
				blas.Copy(s, WS.ws)
			}
			err = f4_no_ir(x, y, z, s)
			for i := 0; i < refinement; i++ {
				blas.Copy(WS.wx, WS.wx2)
				blas.Copy(WS.wy, WS.wy2)
				blas.Copy(WS.wz, WS.wz2)
				blas.Copy(WS.ws, WS.ws2)
				res(x, y, z, s, WS.wx2, WS.wy2, WS.wz2, WS.ws2, W, lmbda)
				f4_no_ir(WS.wx2, WS.wy2, WS.wz2, WS.ws2)
				blas.AxpyFloat(WS.wx2, x, 1.0)
				blas.AxpyFloat(WS.wy2, y, 1.0)
				blas.AxpyFloat(WS.wz2, z, 1.0)
				blas.AxpyFloat(WS.ws2, s, 1.0)
			}
			return
		}
		
		//var mu, sigma, eta float64
		mu = gap / float64(dims.Sum("l", "s") +  len(dims.At("q")))
		sigma, eta = 0.0, 0.0

		for i := 0; i < 2; i++ {
            // Solve
            //
            //     [ 0     ]   [ P  A' G' ]   [ dx        ]
            //     [ 0     ] + [ A  0  0  ] * [ dy        ] = -(1 - eta) * r
            //     [ W'*ds ]   [ G  0  0  ]   [ W^{-1}*dz ]
            //
            //     lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e (i=0)
            //     lmbda o (dz + ds) = -lmbda o lmbda - dsa o dza 
            //                         + sigma*mu*e (i=1) where dsa, dza
            //                         are the solution for i=0. 
 
            // ds = -lmbdasq + sigma * mu * e  (if i is 0)
            //    = -lmbdasq - dsa o dza + sigma * mu * e  (if i is 1), 
            //    where ds, dz are solution for i is 0.
			blas.ScalFloat(ds, 0.0)
			if correction && i == 1 {
				blas.AxpyFloat(ws3, ds, -1.0)
			}
			blas.AxpyFloat(lmbdasq, ds, -1.0, &la.IOpt{"n", dims.Sum("l", "q")})
			ind := dims.At("l")[0]
			ds.Add(sigma*mu, matrix.MakeIndexSet(0, ind, 1)...) 
			for _, m := range dims.At("q") {
				ds.SetIndex(ind, sigma*mu+ds.GetIndex(ind))
				ind += m
			}
			ind2 := ind
			for _, m := range dims.At("s") {
				blas.AxpyFloat(lmbdasq, ds, -1.0, &la.IOpt{"n", m}, &la.IOpt{"incy", m+1},
					&la.IOpt{"offsetx", ind2}, &la.IOpt{"offsety", ind})
				ds.Add(sigma*mu, matrix.MakeIndexSet(ind, ind+m*m, m+1)...)
				ind += m*m
				ind2 += m
			}

			// (dx, dy, dz) := -(1 - eta) * (rx, ry, rz)
			blas.ScalFloat(dx, 0.0)
			blas.AxpyFloat(rx, dx, -1.0+eta)
			blas.ScalFloat(dy, 0.0)
			blas.AxpyFloat(ry, dy, -1.0+eta)
			blas.ScalFloat(dz, 0.0)
			blas.AxpyFloat(rz, dz, -1.0+eta)

			//fmt.Printf("== Calling f4 %d\n", i)
			//fmt.Printf("dx=\n%v\n", dx.ToString("%.17f"))
			//fmt.Printf("ds=\n%v\n", ds.ToString("%.17f"))
			//fmt.Printf("dz=\n%v\n", dz.ToString("%.17f"))
			//fmt.Printf("== Entering f4 %d\n", i)
			err = f4(dx, dy, dz, ds)
			if err != nil {
				if iter == 0 {
					s := fmt.Sprintf("kkt error: %s", err)
					err = errors.New("6: Rank(A) < p or Rank([P; A; G]) < n : "+s)
					return
				} else {
					ind = dims.Sum("l", "q")
					for _, m := range dims.At("s") {
						symm(s, m, ind)
						symm(z, m, ind)
						ind += m*m
					}
					ts,_ = maxStep(s, dims, 0, nil)
					tz,_ = maxStep(z, dims, 0, nil)
					return
				}
			}

			dsdz := sdot(ds, dz, dims, 0)
			if correction && i == 0 {
				blas.Copy(ds, ws3)
				sprod(ws3, dz, dims, 0)
			}

            // Maximum step to boundary.
            // 
            // If i is 1, also compute eigenvalue decomposition of the 's' 
            // blocks in ds, dz.  The eigenvectors Qs, Qz are stored in 
            // dsk, dzk.  The eigenvalues are stored in sigs, sigz. 
			scale2(lmbda, ds, dims, 0, false)
			scale2(lmbda, dz, dims, 0, false)
			if i == 0 {
				ts,_ = maxStep(ds, dims, 0, nil)
				tz,_ = maxStep(dz, dims, 0, nil)
			} else {
				ts,_ = maxStep(ds, dims, 0, sigs)
				tz,_ = maxStep(dz, dims, 0, sigz)
			}
			t := maxvec([]float64{0.0, ts, tz})
			//fmt.Printf("== t=%.17f from %v\n", t, []float64{ts, tz})
			if t == 0.0 {
				step = 1.0
			} else {
				if i == 0 {
					step = math.Min(1.0, 1.0/t)
				} else {
					step = math.Min(1.0, STEP/t)
				}
			}
			if i == 0 {
				m := math.Max(0.0, 1.0 - step + dsdz/gap * (step*step))
				sigma = math.Pow(math.Min(1.0, m), float64(EXPON))
				eta = 0.0
			}
			//fmt.Printf("== step=%.17f sigma=%.17f dsdz=%.17f\n", step, sigma, dsdz)

		}

		blas.AxpyFloat(dx, x, step)
		blas.AxpyFloat(dy, y, step)
		//fmt.Printf("x=\n%v\n", x.ConvertToString())
		//fmt.Printf("y=\n%v\n", y.ConvertToString())
		//fmt.Printf("ds=\n%v\n", ds.ConvertToString())
		//fmt.Printf("dz=\n%v\n", dz.ConvertToString())

        // We will now replace the 'l' and 'q' blocks of ds and dz with 
        // the updated iterates in the current scaling.
        // We also replace the 's' blocks of ds and dz with the factors 
        // Ls, Lz in a factorization Ls*Ls', Lz*Lz' of the updated variables
        // in the current scaling.

        // ds := e + step*ds for nonlinear, 'l' and 'q' blocks.
        // dz := e + step*dz for nonlinear, 'l' and 'q' blocks.
		blas.ScalFloat(ds, step, &la.IOpt{"n", dims.Sum("l", "q")})
		blas.ScalFloat(dz, step, &la.IOpt{"n", dims.Sum("l", "q")})
		ind := dims.At("l")[0]
		is := matrix.MakeIndexSet(0, ind, 1)
		ds.Add(1.0, is...)
		dz.Add(1.0, is...)
		for _, m := range dims.At("q") {
			ds.SetIndex(ind, 1.0+ds.GetIndex(ind))
			dz.SetIndex(ind, 1.0+dz.GetIndex(ind))
			ind += m
		}

        // ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
        // 
        // This replaces the 'l' and 'q' components of ds and dz with the
        // updated variables in the current scaling.  
        // The 's' components of ds and dz are replaced with 
        // 
        // diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2} 
        // diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2} 
		scale2(lmbda, ds, dims, 0, true)
		scale2(lmbda, dz, dims, 0, true)

        // sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
        // sigz := ( e + step*sigz ) ./ lambda for 's' blocks.
        blas.ScalFloat(sigs, step)
        blas.ScalFloat(sigz, step)
        sigs.Add(1.0)
        sigz.Add(1.0)
		sdimsum := dims.Sum("s")
		qdimsum := dims.Sum("l", "q")
		blas.TbsvFloat(lmbda, sigs, &la.IOpt{"n", sdimsum}, &la.IOpt{"k", 0},
			&la.IOpt{"lda", 1}, &la.IOpt{"offseta", qdimsum})
		blas.TbsvFloat(lmbda, sigz, &la.IOpt{"n", sdimsum}, &la.IOpt{"k", 0},
			&la.IOpt{"lda", 1}, &la.IOpt{"offseta", qdimsum})
		
		ind2 := qdimsum; ind3 := 0
		sdims := dims.At("s")
		
		for k := 0; k < len(sdims); k++ {
			m := sdims[k]
			for i := 0; i < m; i++ {
				a := math.Sqrt(sigs.GetIndex(ind3+i))
				blas.ScalFloat(ds, a, &la.IOpt{"offset", ind2+m*i}, &la.IOpt{"n", m})
				a = math.Sqrt(sigz.GetIndex(ind3+i))
				blas.ScalFloat(dz, a, &la.IOpt{"offset", ind2+m*i}, &la.IOpt{"n", m})
			}
			ind2 += m*m
			ind3 += m
		}
		
		err = updateScaling(W, lmbda, ds, dz)

        // Unscale s, z, tau, kappa (unscaled variables are used only to 
        // compute feasibility residuals).
		ind = dims.Sum("l", "q")
		ind2 = ind
		blas.Copy(lmbda, s, &la.IOpt{"n", ind})
		for _, m := range dims.At("s") {
			blas.ScalFloat(s, 0.0, &la.IOpt{"offset", ind2})
			blas.Copy(lmbda, s, &la.IOpt{"offsetx", ind}, &la.IOpt{"offsety", ind2},
				&la.IOpt{"n", m}, &la.IOpt{"incy", m+1})
			ind += m
			ind2 += m*m
		}
		scale(s, W, true, false)
		
		ind = dims.Sum("l", "q")
		ind2 = ind
		blas.Copy(lmbda, z, &la.IOpt{"n", ind})
		for _, m := range dims.At("s") {
			blas.ScalFloat(z, 0.0, &la.IOpt{"offset", ind2})
			blas.Copy(lmbda, z, &la.IOpt{"offsetx", ind}, &la.IOpt{"offsety", ind2},
				&la.IOpt{"n", m}, &la.IOpt{"incy", m+1})
			ind += m
			ind2 += m*m
		}
		scale(z, W, false, true)

		gap = blas.DotFloat(lmbda, lmbda)
		//fmt.Printf("== gap = %.17f\n", gap)
	}
	return
}

// Local Variables:
// tab-width: 4
// End:
