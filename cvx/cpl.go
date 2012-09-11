
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

// ConvexProg is an interface that handles the following functions.
//
// F0() returns a tuple (mnl, x0, error).  
//
//  * mnl is the number of nonlinear inequality constraints.
//  * x0 is a point in the domain of f.
//
// F1(x) returns a tuple (f, Df, error).
//
//  * f is a matrix of size (mnl, 1) containing f(x). 
//  * Df is a matrix of size (mnl, n), containing the derivatives of f at x.
//    Df[k,:] is the transpose of the gradient of fk at x.
//    If x is not in dom f, F1(x) returns (nil, nil, error)
//
// F2(x, z) with z a positive matrix of size (mnl,1), returns a tuple (f, Df, H, error).
//            
//   * f and Df are defined as above.
//   * H is a matrix of size (n,n).  The lower triangular part of H contains the
//     lower triangular part of sum_k z[k] * Hk where Hk is the Hessian of fk at x.
//  
// When F2() is called, it can be assumed that x is dom f. 
//
//
type ConvexProg interface {
	// Returns (mnl, x0) where mln number of nonlinear inequality constraints
	// and x0 is a point in the domain of f.
	F0() (mnl int, x0 *matrix.FloatMatrix, err error)

	// Returns a tuple (f, Df) where f is of size (mnl, 1) containing f(x)
	// Df is matrix of size (mnl, n) containing the derivatives of f at x:
	// Df[k,:] is the transpose of the gradient of fk at x. If x is not in
	// domf, return non-nil error.
	F1(x *matrix.FloatMatrix)(f, Df *matrix.FloatMatrix, err error)
	
	// F(x, z) with z a positive  matrix of size (mnl, 1). Return a tuple
	// (f, Df, H), where f, Df as above. H is matrix of size (n, n).
	F2(x, z *matrix.FloatMatrix)(f, Df, H *matrix.FloatMatrix, err error)
}

//    Solves a convex optimization problem with a linear objective
//
//        minimize    c'*x 
//        subject to  f(x) <= 0
//                    G*x <= h
//                    A*x = b.                      
//
//    f is vector valued, convex and twice differentiable.  The linear 
//    inequalities are with respect to a cone C defined as the Cartesian 
//    product of N + M + 1 cones:
//    
//        C = C_0 x C_1 x .... x C_N x C_{N+1} x ... x C_{N+M}.
//
//    The first cone C_0 is the nonnegative orthant of dimension ml.  The 
//    next N cones are second order cones of dimension mq[0], ..., mq[N-1].
//    The second order cone of dimension m is defined as
//    
//        { (u0, u1) in R x R^{m-1} | u0 >= ||u1||_2 }.
//
//    The next M cones are positive semidefinite cones of order ms[0], ...,
//    ms[M-1] >= 0.  
//
func Cpl(F ConvexProg, c, G, h, A, b *matrix.FloatMatrix, dims *sets.DimensionSet, solopts *SolverOptions) (sol *Solution, err error) {

	const (
		STEP = 0.99
		BETA = 0.5
		ALPHA = 0.01
		EXPON = 3
		MAX_RELAXED_ITERS = 8
	)

	var refinement int

	sol = &Solution{Unknown,
		nil, nil, nil, nil, nil, 
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0}

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
	if solopts.Refinement > 0 {
		refinement = solopts.Refinement
	} else {
		refinement = 1
	}

	solvername := solopts.KKTSolverName
	if len(solvername) == 0 {
		if dims != nil && (len(dims.At("q")) > 0 || len(dims.At("s")) > 0) {
			solvername = "qr"
		} else {
			solvername = "chol2"
		}
	}

	var mnl int
	var x0 *matrix.FloatMatrix

	mnl, x0, err = F.F0()
	if err != nil {
		return
	}

	if x0.Cols() != 1 {
		err = errors.New("'x0' must be matrix with one column")
		return
	}
	if c == nil {
		err = errors.New("'c' must be non nil matrix")
		return
	}
	if ! c.SizeMatch(x0.Size()) {
		err = errors.New(fmt.Sprintf("'c' must be matrix of size (%d,1)", x0.Rows()))
		return 
	}
		
	if h == nil {
		h = matrix.FloatZeros(0, 1)
	}
	if dims == nil {
		dims = sets.DSetNew("l", "q", "s")
		dims.Set("l", []int{h.Rows()})
	}
	if err = checkConeLpDimensions(dims); err != nil {
		return 
	}

	cdim := dims.Sum("l", "q") + dims.SumSquared("s")
	cdim_diag := dims.Sum("l", "q", "s")

	if h.Rows() != cdim {
		err = errors.New(fmt.Sprintf("'h' must be float matrix of size (%d,1)", cdim))
		return 
	}

	if G == nil {
		G = matrix.FloatZeros(0, c.Rows())
	}

	if ! G.SizeMatch(cdim, c.Rows()) {
		estr := fmt.Sprintf("'G' must be of size (%d,%d)", cdim, c.Rows())
		err = errors.New(estr)
		return 
	}

	fG := func(x, y *matrix.FloatMatrix, alpha, beta float64, trans la.Option) error{
		return sgemv(G, x, y, alpha, beta, dims, trans)
	}


	// Check A and set defaults if it is nil
	if A == nil {
		// zeros rows reduces Gemv to vector products
		A = matrix.FloatZeros(0, c.Rows())
	}
	if A.Cols() != c.Rows() {
		estr := fmt.Sprintf("'A' must have %d columns", c.Rows())
		err = errors.New(estr)
		return 
	}

	fA := func(x, y *matrix.FloatMatrix, alpha, beta float64, trans la.Option) error {
		return blas.GemvFloat(A, x, y, alpha, beta, trans)
	}

	if b == nil {
		b = matrix.FloatZeros(0, 1)
	}
	if b.Cols() != 1 {
		err = errors.New("'b' must be matrix with one column.")
		return
	}
	if b.Rows() != A.Rows() {
		err = errors.New(fmt.Sprintf("'b' must have length %d", A.Rows()))
		return
	}

	var factor kktFactor
	var kktsolver kktFactor = nil
	if kktfunc, ok := solvers[solvername]; ok {
		// kkt function returns us problem spesific factor function.
		factor, err = kktfunc(G, dims, A, mnl)
		// solver is 
		kktsolver = func(W *sets.FloatMatrixSet, x, z *matrix.FloatMatrix) (KKTFunc, error) {
			_, Df, H, err := F.F2(x, z)
			if err != nil { return nil, err }
			return factor(W, H, Df)
		}
	} else {
		err = errors.New(fmt.Sprintf("solver '%s' not known", solvername))
		return
	}

	//var x, y, z, s *matrix.FloatMatrix
	//var dx, dy, dz, ds *matrix.FloatMatrix
	//var rx, ry, rznl, rzl *matrix.FloatMatrix
	//var lmbda, lmbdasq *matrix.FloatMatrix

	x := x0.Copy()
	y := b.Copy()
	y.Scale(0.0)
	z := matrix.FloatZeros(mnl+cdim, 1)
	s := matrix.FloatZeros(mnl+cdim, 1)
	ind := mnl+dims.At("l")[0]
	z.Set(1.0, matrix.MakeIndexSet(0, ind, 1)...)
	s.Set(1.0, matrix.MakeIndexSet(0, ind, 1)...)
	for _, m := range dims.At("q") {
		z.Set(1.0, ind)
		s.Set(1.0, ind)
		ind += m
	}
	for _, m := range dims.At("s") {
		iset := matrix.MakeIndexSet(ind, ind+m*m, m+1)
		z.Set(1.0, iset...)
		s.Set(1.0, iset...)
		ind += m*m
	}

	rx := x0.Copy()
	ry := b.Copy()
	rznl := matrix.FloatZeros(mnl, 1)
	rzl := matrix.FloatZeros(cdim, 1)
	dx := x.Copy()
	dy := y.Copy()
	dz := matrix.FloatZeros(mnl+cdim, 1)
	ds := matrix.FloatZeros(mnl+cdim, 1)
	lmbda := matrix.FloatZeros(mnl+cdim_diag, 1)
	lmbdasq := matrix.FloatZeros(mnl+cdim_diag, 1)
	sigs := matrix.FloatZeros(dims.Sum("s"), 1)
	sigz := matrix.FloatZeros(dims.Sum("s"), 1)

	dz2 := matrix.FloatZeros(mnl+cdim, 1)
	ds2 := matrix.FloatZeros(mnl+cdim, 1)

	newx := x.Copy()
	newy := y.Copy()
	newz := matrix.FloatZeros(mnl+cdim, 1)
	news := matrix.FloatZeros(mnl+cdim, 1)
	newrx := x0.Copy()
	newrznl := matrix.FloatZeros(mnl, 1)

	rx0 := x0.Copy()
	ry0 := b.Copy()
	rznl0 := matrix.FloatZeros(mnl, 1)
	rzl0 := matrix.FloatZeros(cdim, 1)
	
	x0, dx0 := x.Copy(), dx.Copy()
	y0, dy0 := y.Copy(), dy.Copy()

	z0 := matrix.FloatZeros(mnl+cdim, 1)
	dz0 := matrix.FloatZeros(mnl+cdim, 1)
	dz20 := matrix.FloatZeros(mnl+cdim, 1)

	s0 := matrix.FloatZeros(mnl+cdim, 1)
	ds0 := matrix.FloatZeros(mnl+cdim, 1)
	ds20 := matrix.FloatZeros(mnl+cdim, 1)
	
	W0 := sets.FloatSetNew("d", "di", "dnl", "dnli", "v", "r", "rti", "beta")
	W0.Set("dnl", matrix.FloatZeros(mnl, 1))
	W0.Set("dnli", matrix.FloatZeros(mnl, 1))
	W0.Set("d", matrix.FloatZeros(dims.At("l")[0], 1))
	W0.Set("di", matrix.FloatZeros(dims.At("l")[0], 1))
	W0.Set("beta", matrix.FloatZeros(len(dims.At("q")), 1))
	for _, n := range dims.At("q") {
		W0.Append("v", matrix.FloatZeros(n, 1))
	}
	for _, n := range dims.At("s") {
		W0.Append("r", matrix.FloatZeros(n, n))
		W0.Append("rti", matrix.FloatZeros(n, n))
	}
	lmbda0 := matrix.FloatZeros(mnl+dims.Sum("l", "q", "s"), 1)
	lmbdasq0 := matrix.FloatZeros(mnl+dims.Sum("l", "q", "s"), 1)

	var f, Df, H *matrix.FloatMatrix = nil, nil, nil
	var ws3, wz3, /*ws2nl, ws2l,*/ wz2l, wz2nl *matrix.FloatMatrix
	var wx, wy, ws, wz, wx2, wy2, wz2, ws2 *matrix.FloatMatrix
	//var sigz, sigs *matrix.FloatMatrix
	var gap, gap0, theta1, theta2, theta3, ts, tz, phi, phi0, mu, sigma, eta float64
	var resx, resy, reszl, resznl, pcost, dcost, dres, pres, relgap float64
	var resx0, /*resy0, reszl0,*/ resznl0, /*pcost0, dcost0,*/ dres0, pres0 float64
	var dsdz, dsdz0, step, step0, dphi, dphi0, sigma0, /*mu0,*/ eta0 float64
	var newresx, newresznl, newgap, newphi float64
	var W *sets.FloatMatrixSet
	var f3 KKTFunc
	
	// Declare fDf and fH here, they bind to Df and H as they are already declared.
	// ??really??
	fDf := func(u, v *matrix.FloatMatrix, alpha, beta float64, trans la.Option) error {
		return blas.GemvFloat(Df, u, v, alpha, beta, trans)
	}
	fH := func(u, v *matrix.FloatMatrix, alpha, beta float64) error {
		return blas.SymvFloat(H, u, v, alpha, beta)
	}

	relaxed_iters := 0
	for iters := 0; iters <= solopts.MaxIter+1; iters++ {

		if refinement != 0 || solopts.Debug {
			f, Df, H, err = F.F2(x, matrix.FloatVector(z.FloatArray()[:mnl]))
		} else {
			f, Df, err = F.F1(x)
		}

		if ! Df.SizeMatch(mnl, c.Rows()) {
			s := fmt.Sprintf("2nd output of F.F2()/F.F1() must matrix of size (%d,%d)",
				mnl, c.Rows())
			err = errors.New(s)
			return
		}

		if refinement != 0 || solopts.Debug {
			if ! H.SizeMatch(c.Rows(), c.Rows()) {
				msg := fmt.Sprintf("3rd output of F.F2() must matrix of size (%d,%d)",
					c.Rows(), c.Rows())
				err = errors.New(msg)
				return
			}
		}

		gap = sdot(s, z, dims, mnl)

		// these are helpers, copies of parts of z,s
		z_mnl := matrix.FloatVector(z.FloatArray()[:mnl])
		z_mnl2 := matrix.FloatVector(z.FloatArray()[mnl:])
		s_mnl := matrix.FloatVector(s.FloatArray()[:mnl])
		s_mnl2 := matrix.FloatVector(s.FloatArray()[mnl:])

		// rx = c + A'*y + Df'*z[:mnl] + G'*z[mnl:]
		blas.Copy(c, rx)
		fA(y, rx, 1.0, 1.0, la.OptTrans)
		fDf(z_mnl, rx, 1.0, 1.0, la.OptTrans)
		fG(z_mnl2, rx, 1.0, 1.0, la.OptTrans)
		resx = math.Sqrt(blas.Dot(rx, rx).Float())

		// rznl = s[:mnl] + f 
		blas.Copy(s_mnl, rznl)
		blas.AxpyFloat(f, rznl, 1.0)
		resznl = blas.Nrm2Float(rznl)

        // rzl = s[mnl:] + G*x - h
        blas.Copy(s_mnl2, rzl)
        blas.AxpyFloat(h, rzl, -1.0)
        fG(x, rzl, 1.0, 1.0, la.OptNoTrans)
        reszl = snrm2(rzl, dims, 0)

		//fmt.Printf("%d: resx=%.9f, resznl=%.9f, reszl=%.9f\n", iters, resx, resznl, reszl)

		// Statistics for stopping criteria
        // pcost = c'*x
        // dcost = c'*x + y'*(A*x-b) + znl'*f(x) + zl'*(G*x-h)
        //       = c'*x + y'*(A*x-b) + znl'*(f(x)+snl) + zl'*(G*x-h+sl) 
        //         - z'*s
        //       = c'*x + y'*ry + znl'*rznl + zl'*rzl - gap
		pcost = blas.DotFloat(c, x)
		dcost = pcost + blas.DotFloat(y, ry) + blas.DotFloat(z_mnl, rznl)
		dcost += sdot(z_mnl2, rzl, dims, 0) - gap
		
		if pcost < 0.0 {
			relgap = gap / -pcost
		} else if dcost > 0.0 {
			relgap = gap / dcost
		} else {
			relgap = math.NaN()
		}
		pres = math.Sqrt(resy*resy + resznl*resznl + reszl*reszl)
		dres = resx
		if iters == 0 {
			resx0 = math.Max(1.0, resx)
			resznl0 = math.Max(1.0, resznl)
			pres0 = math.Max(1.0, pres)
			dres0 = math.Max(1.0, dres)
			gap0 = gap
			theta1 = 1.0/gap0
			theta2 = 1.0/resx0
			theta3 = 1.0/resznl0
		}
		phi = theta1 * gap + theta2 *resx + theta3 * resznl
		pres = pres/pres0
		dres = dres/dres0

		if solopts.ShowProgress {
			if iters == 0 {
				// some headers
				fmt.Printf("% 10s% 12s% 10s% 8s% 7s\n",
					"pcost", "dcost", "gap", "pres", "dres")
			}
            fmt.Printf("%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e\n",
				iters, pcost, dcost, gap, pres, dres)
		}
			
		// Stopping criteria
		if ( pres <= feasTolerance && dres <= feasTolerance &&
			( gap <= absTolerance || (!math.IsNaN(relgap) && relgap <= relTolerance))) ||
			iters == solopts.MaxIter {

			if iters == solopts.MaxIter {
				s := "Terminated (maximum number of iterations reached)"
				if solopts.ShowProgress {
					fmt.Printf(s + "\n")
				}
				err = errors.New(s)
				sol.Status = Unknown
			} else {
				err = nil
				sol.Status = Optimal
			}
			sol.Result = sets.FloatSetNew("x", "y", "znl", "zl", "snl", "sl")
			sol.Result.Set("x", x)
			sol.Result.Set("y", y)
			sol.Result.Set("znl", matrix.FloatVector(z.FloatArray()[:mnl]))
			sol.Result.Set("zl",  matrix.FloatVector(z.FloatArray()[mnl:]))
			sol.Result.Set("sl",  matrix.FloatVector(s.FloatArray()[mnl:]))
			sol.Result.Set("snl", matrix.FloatVector(s.FloatArray()[:mnl]))
			sol.Gap = gap
			sol.RelativeGap = relgap
			sol.PrimalObjective = pcost
			sol.DualObjective = dcost
			sol.PrimalInfeasibility = pres
			sol.DualInfeasibility = dres
			sol.PrimalSlack = -ts
			sol.DualSlack = -tz
			return
		}

        // Compute initial scaling W: 
        //
        //     W * z = W^{-T} * s = lambda.
        //
        // lmbdasq = lambda o lambda 
        if iters == 0 {
            W, _ = computeScaling(s, z, lmbda, dims, mnl)
		}
        ssqr(lmbdasq, lmbda, dims, mnl)

        // f3(x, y, z) solves
        //
        //     [ H   A'  GG'*W^{-1} ] [ ux ]   [ bx ]
        //     [ A   0   0          ] [ uy ] = [ by ].
        //     [ GG  0  -W'         ] [ uz ]   [ bz ]
        //
        // On entry, x, y, z contain bx, by, bz.
        // On exit, they contain ux, uy, uz.
        f3, err = kktsolver(W, x, z_mnl)
		if err != nil {
			// ?? z_mnl is really copy of z[:mnl] ... should we copy here back to z??
			singular_kkt_matrix := false
			if iters == 0 {
				err = errors.New("Rank(A) < p or Rank([H(x); A; Df(x); G] < n")
				return
			} else if relaxed_iters > 0 && relaxed_iters < MAX_RELAXED_ITERS {
				// The arithmetic error may be caused by a relaxed line 
				// search in the previous iteration.  Therefore we restore 
				// the last saved state and require a standard line search. 
				phi, gap = phi0, gap0
				mu = gap / float64(mnl + dims.Sum("l", "s") + len(dims.At("q")))
				blas.Copy(W0.At("dnl")[0],  W.At("dnl")[0])
				blas.Copy(W0.At("dnli")[0], W.At("dnli")[0])
				blas.Copy(W0.At("d")[0],    W.At("d")[0])
				blas.Copy(W0.At("di")[0],   W.At("di")[0])
				blas.Copy(W0.At("beta")[0], W.At("beta")[0])
				for k, _ := range dims.At("q") {
					blas.Copy(W0.At("v")[k], W.At("v")[k])
				}
				for k, _ := range dims.At("s") {
					blas.Copy(W0.At("r")[k], W.At("r")[k])
					blas.Copy(W0.At("rti")[k], W.At("rti")[k])
				}
				blas.Copy(x0, x)
				blas.Copy(y0, y)
				blas.Copy(s0, s)
				blas.Copy(z0, z)
				blas.Copy(lmbda0, lmbda)
				blas.Copy(lmbdasq0, lmbdasq) // ???
				blas.Copy(rx0, rx)
				blas.Copy(ry0, ry)
				resx = math.Sqrt(blas.DotFloat(rx, rx))
				blas.Copy(rznl0, rznl)
				blas.Copy(rzl0, rzl)
				resznl = blas.Nrm2Float(rznl)

				relaxed_iters = -1

				// How about z_mnl here???
				f3, err = kktsolver(W, x, z_mnl)
				if err != nil {
					singular_kkt_matrix = true
				}
			} else {
				singular_kkt_matrix = true
			}


			if singular_kkt_matrix {
				msg := "Terminated (singular KKT matrix)."
				if solopts.ShowProgress {
					fmt.Printf(msg + "\n")
				}
				zl := matrix.FloatVector(z.FloatArray()[mnl:])
				sl := matrix.FloatVector(s.FloatArray()[mnl:])
				ind := dims.Sum("l", "q")
				for _, m := range dims.At("s") {
					symm(sl, m, ind)
					symm(zl, m, ind)
					ind += m*m
				}
				ts, _ = maxStep(s, dims, mnl, nil)
				tz, _ = maxStep(z, dims, mnl, nil)

				err = errors.New(msg)
				sol.Status = Unknown
				sol.Result = sets.FloatSetNew("x", "y", "znl", "zl", "snl", "sl")
				sol.Result.Set("x", x)
				sol.Result.Set("y", y)
				sol.Result.Set("znl", matrix.FloatVector(z.FloatArray()[:mnl]))
				sol.Result.Set("zl",  zl)
				sol.Result.Set("sl",  sl)
				sol.Result.Set("snl", matrix.FloatVector(s.FloatArray()[:mnl]))
				sol.Gap = gap
				sol.RelativeGap = relgap
				sol.PrimalObjective = pcost
				sol.DualObjective = dcost
				sol.PrimalInfeasibility = pres
				sol.DualInfeasibility = dres
				sol.PrimalSlack = -ts
				sol.DualSlack = -tz
				return
			}
		}

        // f4_no_ir(x, y, z, s) solves
        // 
        //     [ 0     ]   [ H   A'  GG' ] [ ux        ]   [ bx ]
        //     [ 0     ] + [ A   0   0   ] [ uy        ] = [ by ]
        //     [ W'*us ]   [ GG  0   0   ] [ W^{-1}*uz ]   [ bz ]
        //
        //     lmbda o (uz + us) = bs.
        //
        // On entry, x, y, z, x, contain bx, by, bz, bs.
        // On exit, they contain ux, uy, uz, us.

        if iters == 0 {
            ws3 = matrix.FloatZeros(mnl + cdim, 1)
            wz3 = matrix.FloatZeros(mnl + cdim, 1)
		}

        f4_no_ir := func(x, y, z, s *matrix.FloatMatrix) (err error) {
			// Solve 
            //
            //     [ H  A'  GG'  ] [ ux        ]   [ bx                    ]
            //     [ A  0   0    ] [ uy        ] = [ by                    ]
            //     [ GG 0  -W'*W ] [ W^{-1}*uz ]   [ bz - W'*(lmbda o\ bs) ]
            //
            //     us = lmbda o\ bs - uz.
            
			err = nil
            // s := lmbda o\ s 
            //    = lmbda o\ bs
            sinv(s, lmbda, dims, mnl)

            // z := z - W'*s 
            //    = bz - W' * (lambda o\ bs)
            blas.Copy(s, ws3)
            scale(ws3, W, true, false)
            blas.AxpyFloat(ws3, z, -1.0)

            // Solve for ux, uy, uz
            f3(x, y, z)

            // s := s - z 
            //    = lambda o\ bs - z.
            blas.AxpyFloat(z, s, -1.0)
			return
		}

        if iters == 0 {
            wz2nl = matrix.FloatZeros(mnl, 1)
            wz2l = matrix.FloatZeros(cdim, 1)
		}

		res := func(ux, uy, uz, us, vx, vy, vz, vs *matrix.FloatMatrix)(err error) {

            // Evaluates residuals in Newton equations:
            //
            //     [ vx ]     [ 0     ]   [ H  A' GG' ] [ ux        ]
            //     [ vy ] -=  [ 0     ] + [ A  0  0   ] [ uy        ]
            //     [ vz ]     [ W'*us ]   [ GG 0  0   ] [ W^{-1}*uz ]
            //
            //     vs -= lmbda o (uz + us).
			err = nil
            // vx := vx - H*ux - A'*uy - GG'*W^{-1}*uz
            fH(ux, vx, -1.0, 1.0)
            fA(uy, vx, -1.0, 1.0, la.OptTrans) 
            blas.Copy(uz, wz3)
            scale(wz3, W, false, true)
			wz3_nl := matrix.FloatVector(wz3.FloatArray()[:mnl])
			wz3_l := matrix.FloatVector(wz3.FloatArray()[mnl:])
            fDf(wz3_nl, vx, -1.0, 1.0, la.OptTrans)
            fG(wz3_l, vx, -1.0, 1.0, la.OptTrans) 

            // vy := vy - A*ux 
            fA(ux, vy, -1.0, 1.0, la.OptNoTrans)

            // vz := vz - W'*us - GG*ux 
            fDf(ux, wz2nl, 1.0, 0.0, la.OptNoTrans)
            blas.AxpyFloat(wz2nl, vz, -1.0)
            fG(ux, wz2l, 1.0, 0.0, la.OptNoTrans)
            blas.AxpyFloat(wz2l, vz, -1.0, &la.IOpt{"offsety", mnl})
            blas.Copy(us, ws3) 
            scale(ws3, W, true, false)
            blas.AxpyFloat(ws3, vz, -1.0)

            // vs -= lmbda o (uz + us)
            blas.Copy(us, ws3)
            blas.AxpyFloat(uz, ws3, 1.0)
            sprod(ws3, lmbda, dims, mnl, &la.SOpt{"diag", "D"})
            blas.AxpyFloat(ws3, vs, -1.0)
			return 
		}

        // f4(x, y, z, s) solves the same system as f4_no_ir, but applies
        // iterative refinement.
		
		if iters == 0 {
			if refinement > 0 || solopts.Debug {
				wx = x.Copy()
				wy = y.Copy()
				wz = z.Copy()
				ws = s.Copy()
			}
			if refinement > 0 {
				wx2 = x.Copy()
				wy2 = y.Copy()
				wz2 = matrix.FloatZeros(mnl+cdim, 1)
				ws2 = matrix.FloatZeros(mnl+cdim, 1)
			}
		}

		f4 := func(x, y, z, s *matrix.FloatMatrix)(err error) {
			if refinement > 0 || solopts.Debug {
				blas.Copy(x, wx)
				blas.Copy(y, wy)
				blas.Copy(z, wz)
				blas.Copy(s, ws)
			}
			err = f4_no_ir(x, y, z, s)
			for i := 0; i < refinement; i++ {
				blas.Copy(wx, wx2)
				blas.Copy(wy, wy2)
				blas.Copy(wz, wz2)
				blas.Copy(ws, ws2)
				res(x, y, z, s, wx2, wy2, wz2, ws2)
				err = f4_no_ir(wx2, wy2, wz2, ws2)
				blas.AxpyFloat(wx2, x, 1.0)
				blas.AxpyFloat(wy2, y, 1.0)
				blas.AxpyFloat(wz2, z, 1.0)
				blas.AxpyFloat(ws2, s, 1.0)
			}
			if solopts.Debug {
				res(x, y,z, s, wx, wy, wz, ws)
				fmt.Printf("KKT residuals:\n")
			}
			return
		}

		sigma, eta = 0.0, 0.0
		for i := 0; i < 2; i++ {
            // Solve
            //
            //     [ 0     ]   [ H  A' GG' ] [ dx        ]
            //     [ 0     ] + [ A  0  0   ] [ dy        ] = -(1 - eta)*r  
            //     [ W'*ds ]   [ GG 0  0   ] [ W^{-1}*dz ]
            //
            //     lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e.
            //

			mu = gap / float64(mnl + dims.Sum("l", "s") + len(dims.At("q")))
			blas.ScalFloat(ds, 0.0)
			blas.AxpyFloat(lmbdasq, ds, -1.0, &la.IOpt{"n", mnl+dims.Sum("l", "s")})

			ind = mnl + dims.At("l")[0]
			iset := matrix.MakeIndexSet(0, ind, 1)
			ds.Add(sigma*mu, iset...)
			for _, m := range dims.At("q") {
				ds.Add(sigma*mu, ind)
				ind += m
			}
			ind2 := ind
			for _, m := range dims.At("s") {
				blas.AxpyFloat(lmbdasq, ds, -1.0, &la.IOpt{"n", m},	&la.IOpt{"offsetx", ind2},
					&la.IOpt{"offsety", ind}, &la.IOpt{"incy", m+1})
				ds.Add(sigma*mu, matrix.MakeIndexSet(ind, ind+m*m, m+1)...)
				ind += m*m
				ind2 += m
			}
			
			dx.Scale(0.0)
			blas.AxpyFloat(rx, dx, -1.0+eta)
			dy.Scale(0.0)
			blas.AxpyFloat(ry, dy, -1.0+eta)
			dz.Scale(0.0)
			blas.AxpyFloat(rznl, dz, -1.0+eta)
			blas.AxpyFloat(rzl, dz, -1.0+eta, &la.IOpt{"offsety", mnl})
			//fmt.Printf("dx=\n%v\n", dx.ToString("%.7f"))
			//fmt.Printf("dz=\n%v\n", dz.ToString("%.7f"))
			//fmt.Printf("ds=\n%v\n", ds.ToString("%.7f"))

			err = f4(dx, dy, dz, ds)

			if err != nil {
				if iters == 0 {
					s := fmt.Sprintf("Rank(A) < p or Rank([H(x); A; Df(x); G] < n (%s)", err)
					err = errors.New(s)
					return
				}
				msg := "Terminated (singular KKT matrix)."
				if solopts.ShowProgress {
					fmt.Printf(msg + "\n")
				}
				zl := matrix.FloatVector(z.FloatArray()[mnl:])
				sl := matrix.FloatVector(s.FloatArray()[mnl:])
				ind := dims.Sum("l", "q")
				for _, m := range dims.At("s") {
					symm(sl, m, ind)
					symm(zl, m, ind)
					ind += m*m
				}
				ts, _ = maxStep(s, dims, mnl, nil)
				tz, _ = maxStep(z, dims, mnl, nil)

				err = errors.New(msg)
				sol.Status = Unknown
				sol.Result = sets.FloatSetNew("x", "y", "znl", "zl", "snl", "sl")
				sol.Result.Set("x", x)
				sol.Result.Set("y", y)
				sol.Result.Set("znl", matrix.FloatVector(z.FloatArray()[:mnl]))
				sol.Result.Set("zl",  zl)
				sol.Result.Set("sl",  sl)
				sol.Result.Set("snl", matrix.FloatVector(s.FloatArray()[:mnl]))
				sol.Gap = gap
				sol.RelativeGap = relgap
				sol.PrimalObjective = pcost
				sol.DualObjective = dcost
				sol.PrimalInfeasibility = pres
				sol.DualInfeasibility = dres
				sol.PrimalSlack = -ts
				sol.DualSlack = -tz
				return
			}
				
			//fmt.Printf("dx=\n%v\n", dx.ToString("%.7f"))
			//fmt.Printf("dz=\n%v\n", dz.ToString("%.7f"))

            // Inner product ds'*dz and unscaled steps are needed in the 
            // line search.
            dsdz = sdot(ds, dz, dims, mnl)
            blas.Copy(dz, dz2)
            scale(dz2, W, false, true)
            blas.Copy(ds, ds2)
            scale(ds2, W, true, false)

            // Maximum steps to boundary. 
            // 
            // Also compute the eigenvalue decomposition of 's' blocks in 
            // ds, dz.  The eigenvectors Qs, Qz are stored in ds, dz.
            // The eigenvalues are stored in sigs, sigz.

            scale2(lmbda, ds, dims, mnl, false)
            ts, _ = maxStep(ds, dims, mnl, sigs)
            scale2(lmbda, dz, dims, mnl, false)
            tz, _ = maxStep(dz, dims, mnl, sigz)
            t := maxvec([]float64{0.0, ts, tz})
            if t == 0 {
                step = 1.0
			} else {
                step = math.Min(1.0, STEP / t)
			}

            //fmt.Printf("%d: ts=%.7f, tz=%.7f, t=%.7f, step=%.7f" , iters, ts, tz, t, step)

			var newDf, newf *matrix.FloatMatrix

            // Backtrack until newx is in domain of f.
			backtrack := true
			//fmt.Printf("backtracking ...\n")
			for backtrack {
				blas.Copy(x, newx)
				blas.AxpyFloat(dx, newx, step)
				newf, newDf, err = F.F1(newx)
				if newf != nil {
					backtrack = false
				} else {
					step *= BETA
				}
			}

            // Merit function 
            //
            //     phi = theta1 * gap + theta2 * norm(rx) + 
            //         theta3 * norm(rznl)
            //
            // and its directional derivative dphi.

			phi = theta1 * gap + theta2 * resx + theta3 * resznl
			if i == 0 {
				dphi = -phi
			} else {
				dphi = -theta1*(1-sigma)*gap - theta2*(1-eta)*resx - theta3*(1-eta)*resznl
			}

			var newfDf func(x, y *matrix.FloatMatrix, a, b float64, trans la.Option)(error)

			// Line search
			backtrack = true
			//fmt.Printf("start line search ...\n")
			for backtrack {

				blas.Copy(x, newx)
				blas.AxpyFloat(dx, newx, step)
				blas.Copy(y, newy)
				blas.AxpyFloat(dy, newy, step)
				blas.Copy(z, newz)
				blas.AxpyFloat(dz2, newz, step)
				blas.Copy(s, news)
				blas.AxpyFloat(ds2, news, step)
				
				newf, newDf, err = F.F1(newx)
				if newDf == nil || ! newDf.SizeMatch(mnl, c.Rows()) {
					msg := fmt.Sprintf("2nd output argument of F.F1() must be of size"+
						" (%d,%d), has size (%d,%d)", mnl, c.Rows(), newDf.Rows(), newDf.Cols())
					err = errors.New(msg)
					return
				}
				newfDf = func(u, v *matrix.FloatMatrix, a, b float64, trans la.Option)(error) {
					return blas.GemvFloat(newDf, u, v, a, b, trans)
				}
				
				//fmt.Printf("news=\n%v\n", news.ToString("%.7f"))
				//fmt.Printf("newf=\n%v\n", newf.ToString("%.7f"))

                // newrx = c + A'*newy + newDf'*newz[:mnl] + G'*newz[mnl:]
				newz_mnl := matrix.FloatVector(newz.FloatArray()[:mnl])
				newz_ml := matrix.FloatVector(newz.FloatArray()[mnl:])
				blas.Copy(c, newrx)
				fA(newy, newrx, 1.0, 1.0, la.OptTrans)
				newfDf(newz_mnl, newrx, 1.0, 1.0, la.OptTrans)
				fG(newz_ml, newrx, 1.0, 1.0, la.OptTrans)
				newresx = math.Sqrt(blas.DotFloat(newrx, newrx))
				
                // newrznl = news[:mnl] + newf 
				news_mnl := matrix.FloatVector(news.FloatArray()[:mnl])
				//news_ml := matrix.FloatVector(news.FloatArray()[mnl:])
				blas.Copy(news_mnl, newrznl)
				blas.AxpyFloat(newf, newrznl, 1.0)
				newresznl = blas.Nrm2Float(newrznl)
				
				newgap = (1.0 - (1.0-sigma)*step)*gap + step*step*dsdz
				newphi = theta1*newgap + theta2*newresx + theta3*newresznl

				//fmt.Printf("theta1=%.7f theta2=%.7f theta3=%.7f\n", theta1, theta2, theta3)
				//fmt.Printf("newgap=%.7f, newphi=%.7f nresx=%.7f nresznl=%.7f\n", newgap, newphi, newresx, newresznl)
				if i == 0 {
					if newgap <= (1.0-ALPHA*step)*gap &&
						(relaxed_iters > 0 && relaxed_iters < MAX_RELAXED_ITERS ||
						newphi <= phi + ALPHA*step*dphi) {
						backtrack = false
						sigma = math.Min(newgap/gap, math.Pow((newgap/gap), EXPON))
						//fmt.Printf("break 1: sigma=%.7f\n", sigma)
						eta = 0.0
					} else {
						step *= BETA
					}
				} else {
					if relaxed_iters == -1 || (relaxed_iters == 0 && MAX_RELAXED_ITERS == 0) {
                        // Do a standard line search.
						if newphi <= phi + ALPHA*step*dphi {
							relaxed_iters = 0
							backtrack = false
							//fmt.Printf("break 2 : newphi=%.7f\n", newphi)
						} else {
							step *= BETA
						}
					} else if relaxed_iters == 0 && relaxed_iters < MAX_RELAXED_ITERS {
						if newphi <= phi +ALPHA*step*dphi {
                            // Relaxed l.s. gives sufficient decrease.
							relaxed_iters = 0
						} else {
                            // Save state.
                            phi0, dphi0, gap0 = phi, dphi, gap
                            step0 = step
								
							blas.Copy(W.At("dnl")[0],  W0.At("dnl")[0])
							blas.Copy(W.At("dnli")[0], W0.At("dnli")[0])
							blas.Copy(W.At("d")[0],    W0.At("d")[0])
							blas.Copy(W.At("di")[0],   W0.At("di")[0])
							blas.Copy(W.At("beta")[0], W0.At("beta")[0])
							for k, _ := range dims.At("q") {
								blas.Copy(W.At("v")[k], W0.At("v")[k])
							}
							for k, _ := range dims.At("s") {
								blas.Copy(W.At("r")[k],   W0.At("r")[k])
								blas.Copy(W.At("rti")[k], W0.At("rti")[k])
							}
							blas.Copy(x, x0)
							blas.Copy(y, y0)
							blas.Copy(dx, dx0)
							blas.Copy(dy, dy0)
							blas.Copy(s, s0)
							blas.Copy(z, z0)
							blas.Copy(ds, ds0)
							blas.Copy(dz, dz0)
							blas.Copy(ds2, ds20)
							blas.Copy(dz2, dz20)
							blas.Copy(lmbda, lmbda0)
							blas.Copy(lmbdasq, lmbdasq0) // ???
							blas.Copy(rx, rx0)
							blas.Copy(ry, ry0)
							blas.Copy(rznl, rznl0)
							blas.Copy(rzl, rzl0)
							dsdz0 = dsdz
							sigma0, eta0 = sigma, eta
							relaxed_iters = 1
						}
						backtrack = false
						//fmt.Printf("break 3 : newphi=%.7f\n", newphi)

					} else if relaxed_iters >= 0 && relaxed_iters < MAX_RELAXED_ITERS &&
						MAX_RELAXED_ITERS > 0 {
						if newphi <= phi0 + ALPHA*step0*dphi0 {
                            // Relaxed l.s. gives sufficient decrease.
							relaxed_iters = 0
						} else {
                            // Relaxed line search 
							relaxed_iters += 1
						}
						backtrack = false
						//fmt.Printf("break 4 : newphi=%.7f\n", newphi)

					} else if relaxed_iters == MAX_RELAXED_ITERS && MAX_RELAXED_ITERS > 0 {
						if newphi <= phi0 + ALPHA*step0*dphi0 {
                            // Series of relaxed line searches ends 
                            // with sufficient decrease w.r.t. phi0.
							backtrack = false
							relaxed_iters = 0
							//fmt.Printf("break 5 : newphi=%.7f\n", newphi)
						} else if newphi >= phi0 {
                            // Resume last saved line search 
                            phi, dphi, gap = phi0, dphi0, gap0
                            step = step0
							blas.Copy(W0.At("dnl")[0],  W.At("dnl")[0])
							blas.Copy(W0.At("dnli")[0], W.At("dnli")[0])
							blas.Copy(W0.At("d")[0],    W.At("d")[0])
							blas.Copy(W0.At("di")[0],   W.At("di")[0])
							blas.Copy(W0.At("beta")[0], W.At("beta")[0])
							for k, _ := range dims.At("q") {
								blas.Copy(W0.At("v")[k], W.At("v")[k])
							}
							for k, _ := range dims.At("s") {
								blas.Copy(W0.At("r")[k],   W.At("r")[k])
								blas.Copy(W0.At("rti")[k], W.At("rti")[k])
							}
							blas.Copy(x, x0)
							blas.Copy(y, y0)
							blas.Copy(dx, dx0)
							blas.Copy(dy, dy0)
							blas.Copy(s, s0)
							blas.Copy(z, z0)
							blas.Copy(ds2, ds20)
							blas.Copy(dz2, dz20)
							blas.Copy(lmbda, lmbda0)
							blas.Copy(lmbdasq, lmbdasq0) // ???
							blas.Copy(rx, rx0)
							blas.Copy(ry, ry0)
							blas.Copy(rznl, rznl0)
							blas.Copy(rzl, rzl0)
							dsdz = dsdz0
							sigma, eta = sigma0, eta0
							relaxed_iters = -1

						} else if newphi <= phi + ALPHA*step*dphi {
                            // Series of relaxed line searches ends 
                            // with sufficient decrease w.r.t. phi0.
							backtrack = false
							relaxed_iters = -1
							//fmt.Printf("break 5 : newphi=%.7f\n", newphi)
						}
					}
				}
			} // end of line search

			//fmt.Printf("eol ds=\n%v\n", ds.ToString("%.7f"))
			//fmt.Printf("eol dz=\n%v\n", dz.ToString("%.7f"))

		} // end for [0,1]

		// Update x, y
		blas.AxpyFloat(dx, x, step)
		blas.AxpyFloat(dy, y, step)
		//fmt.Printf("update x=\n%v\n", x.ToString("%.7f"))
		//fmt.Printf("z=\n%v\n", z.ToString("%.7f"))

		// Replace nonlinear, 'l' and 'q' blocks of ds and dz with the 
		// updated variables in the current scaling.
		// Replace 's' blocks of ds and dz with the factors Ls, Lz in a
		// factorization Ls*Ls', Lz*Lz' of the updated variables in the 
		// current scaling.
		
		// ds := e + step*ds for nonlinear, 'l' and 'q' blocks.
		// dz := e + step*dz for nonlinear, 'l' and 'q' blocks.
		blas.ScalFloat(ds, step, &la.IOpt{"n", mnl + dims.Sum("l", "q")})
		blas.ScalFloat(dz, step, &la.IOpt{"n", mnl + dims.Sum("l", "q")})
		ind := mnl + dims.At("l")[0]
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
		scale2(lmbda, ds, dims, mnl, true)
		scale2(lmbda, dz, dims, mnl, true)

        // sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
        // sigz := ( e + step*sigz ) ./ lambda for 's' blocks.
        blas.ScalFloat(sigs, step)
        blas.ScalFloat(sigz, step)
        sigs.Add(1.0)
        sigz.Add(1.0)
		sdimsum := dims.Sum("s")
		qdimsum := dims.Sum("l", "q")
		blas.TbsvFloat(lmbda, sigs, &la.IOpt{"n", sdimsum}, &la.IOpt{"k", 0},
			&la.IOpt{"lda", 1}, &la.IOpt{"offseta", mnl+qdimsum})
		blas.TbsvFloat(lmbda, sigz, &la.IOpt{"n", sdimsum}, &la.IOpt{"k", 0},
			&la.IOpt{"lda", 1}, &la.IOpt{"offseta", mnl+qdimsum})
		

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
		ind = mnl + dims.Sum("l", "q")
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
		
		ind = mnl + dims.Sum("l", "q")
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

	}
	return
}


// Local Variables:
// tab-width: 4
// End:

