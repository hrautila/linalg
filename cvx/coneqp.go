
package cvx

import (
	la_ "github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/matrix"
	"errors"
	"fmt"
	"math"
)

type InitVals struct {
	X, Y, S, Z *matrix.FloatMatrix
}

//    Solves a pair of primal and dual convex quadratic cone programs
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
//    The inequalities are with respect to a cone C defined as the Cartesian
//    product of N + M + 1 cones:
//    
//        C = C_0 x C_1 x .... x C_N x C_{N+1} x ... x C_{N+M}.
//
//    The first cone C_0 is the nonnegative orthant of dimension ml.  
//    The next N cones are 2nd order cones of dimension mq[0], ..., mq[N-1].
//    The second order cone of dimension m is defined as
//    
//        { (u0, u1) in R x R^{m-1} | u0 >= ||u1||_2 }.
//
//    The next M cones are positive semidefinite cones of order ms[0], ...,
//    ms[M-1] >= 0.  
//
func ConeQp(P, q, G, h, A, b *matrix.FloatMatrix, dims *DimensionSet, solopts *SolverOptions) (sol *Solution, err error) {

	var initvals *InitVals = nil

	err = nil
	EXPON := 3
	STEP := 0.99

	sol = &Solution{Unknown,
		nil, nil, nil, nil,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0}

	var kktsolver func(*FloatMatrixSet)(kktFunc, error) = nil
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

	solvername := solopts.KKTSolverName
	if len(solvername) == 0 {
		if dims != nil && (len(dims.At("q")) > 0 || len(dims.At("s")) > 0) {
			solvername = "qr"
			//kktsolver = solvers["qr"]
		} else {
			solvername = "chol2"
			//kktsolver = solvers["chol2"]
		}
	}

	if q == nil || q.Cols() != 1 {
		err = errors.New("'q' must be non-nil matrix with one column")
		return
	}
	if P == nil || P.Rows() != q.Rows() || P.Cols() != q.Rows() {
		err = errors.New(fmt.Sprintf("'P' must be non-nil matrix of size (%d, %d)",
			q.Rows(), q.Rows()))
		return
	}
	fP := func(x, y *matrix.FloatMatrix, alpha, beta float64) error{
		return blas.SymvFloat(P, x, y, alpha, beta)
	}

	if h == nil {
		h = matrix.FloatZeros(0, 1)
	}
	if h.Cols() != 1 {
		err = errors.New("'h' must be non-nil matrix with one column")
		return
	}
	if dims == nil {
		dims = DSetNew("l", "q", "s")
		dims.Set("l", []int{h.Rows()})
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

	if G != nil && !G.SizeMatch(cdim, q.Rows()) {
		estr := fmt.Sprintf("'G' must be of size (%d,%d)", cdim, q.Rows())
		err = errors.New(estr)
		return 
	}
	fG := func(x, y *matrix.FloatMatrix, alpha, beta float64, opts ...la_.Option) error{
		return Sgemv(G, x, y, alpha, beta, dims, opts...)
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

	fA := func(x, y *matrix.FloatMatrix, alpha, beta float64, opts ...la_.Option) error {
		return blas.GemvFloat(A, x, y, alpha, beta, opts...)
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

    // kktsolver(W) returns a routine for solving 3x3 block KKT system 
    //
    //     [ 0   A'  G'*W^{-1} ] [ ux ]   [ bx ]
    //     [ A   0   0         ] [ uy ] = [ by ].
    //     [ G   0   -W'       ] [ uz ]   [ bz ]
	var factor kktFactor
	if kkt, ok := solvers[solvername]; ok {
		if b.Rows() > q.Rows()  {
			err = errors.New("Rank(A) < p or Rank[G; A] < n")
			return
		}
		if kkt == nil {
			err = errors.New(fmt.Sprintf("solver '%s' not yet implemented", solvername))
			return
		}
		// kkt function returns us problem spesific factor function.
		factor, err = kkt(G, dims, A, 0)
		// solver is 
		kktsolver = func(W *FloatMatrixSet) (kktFunc, error) {
			return factor(W, P, nil)
		}
	} else {
		err = errors.New(fmt.Sprintf("solver '%s' not known", solvername))
		return
	}

	ws3 := matrix.FloatZeros(cdim, 1)
	wz3 := matrix.FloatZeros(cdim, 1)

	// 
	res := func(ux, uy, uz, us, vx, vy, vz, vs *matrix.FloatMatrix, W *FloatMatrixSet, lmbda *matrix.FloatMatrix) (err error) {
        // Evaluates residual in Newton equations:
        // 
        //      [ vx ]    [ vx ]   [ 0     ]   [ P  A'  G' ]   [ ux        ]
        //      [ vy ] := [ vy ] - [ 0     ] - [ A  0   0  ] * [ uy        ]
        //      [ vz ]    [ vz ]   [ W'*us ]   [ G  0   0  ]   [ W^{-1}*uz ]
        //
        //      vs := vs - lmbda o (uz + us).

        // vx := vx - P*ux - A'*uy - G'*W^{-1}*uz
		fP(ux, vx, -1.0, 1.0)
		fA(uy, vx, -1.0, 1.0, la_.OptTrans)
		blas.Copy(uz, wz3)
		Scale(wz3, W, true, false)
		fG(wz3, vx, -1.0, 1.0, la_.OptTrans)
        // vy := vy - A*ux
        fA(ux, vy, -1.0, 1.0)

        // vz := vz - G*ux - W'*us
        fG(ux, vz, -1.0, 1.0)
        blas.Copy(us, ws3)
        Scale(ws3, W, true, false)
        blas.AxpyFloat(ws3, vz, -1.0)
 
        // vs := vs - lmbda o (uz + us)
        blas.Copy(us, ws3)
        blas.AxpyFloat(uz, ws3, 1.0)
        Sprod(ws3, lmbda, dims, 0, la_.OptDiag)
        blas.AxpyFloat(ws3, vs, -1.0)
		return 
	}

	resx0 := math.Max(1.0, math.Sqrt(blas.Dot(q,q).Float()))
	resy0 := math.Max(1.0, math.Sqrt(blas.Dot(b,b).Float()))
	resz0 := math.Max(1.0, Snrm2(h, dims, 0))
	fmt.Printf("resx0: %.9f, resy0: %.9f, resz0: %9.f\n", resx0, resy0, resz0)

	var x, y, z, s, dx, dy, ds, dz, rx, ry, rz *matrix.FloatMatrix
	var lmbda, lmbdasq, sigs, sigz *matrix.FloatMatrix
	var W *FloatMatrixSet
	var f, f3 kktFunc
	var resx, resy, resz, step, sigma, mu, eta float64
	var gap, pcost, dcost, relgap, pres, dres, f0 float64

	if cdim == 0 {
		// Solve
		//
		//     [ P  A' ] [ x ]   [ -q ]
		//     [       ] [   ] = [    ].
		//     [ A  0  ] [ y ]   [  b ]
		//
		Wtmp := FloatSetNew("d", "di", "beta", "v", "r", "rti")
		Wtmp.Set("d", matrix.FloatZeros(0, 1))
		Wtmp.Set("di", matrix.FloatZeros(0, 1))
		f3, err = kktsolver(Wtmp)
		if err != nil {
			err = errors.New("Rank(A) < p or Rank(([P; A; G;]) < n")
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
		fA(y, rx, 1.0, 1.0, la_.OptTrans)
		dres = math.Sqrt(blas.DotFloat(rx, rx)/resx0)
		
		ry = b.Copy()
		fA(x, ry, 1.0, -1.0)
		pres = math.Sqrt(blas.DotFloat(ry, ry)/resy0)

		relgap = 0.0
		if pcost == 0.0 {
			relgap = math.NaN()
		}

		sol.X = x
		sol.Y = y
		sol.S = matrix.FloatZeros(0, 1)
		sol.Z = matrix.FloatZeros(0, 1)
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
		W = FloatSetNew("d", "di", "v", "beta", "r", "rti")
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
			err = errors.New("Rank(A) < p or Rank([P; G; A]) < n")
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
			err = errors.New("Rank(A) < p or Rank([P; G; A]) < n")
			return 
		}
		s = z.Copy()
		blas.ScalFloat(s, -1.0)

		nrms = Snrm2(s, dims, 0)
		ts,_ = MaxStep(s, dims, 0, nil)
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

		nrmz = Snrm2(z, dims, 0)
		tz,_ = MaxStep(z, dims, 0, nil)
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
		if initvals.X == nil {
			blas.Copy(initvals.X, x)
		} else {
			blas.ScalFloat(x, 0.0)
		}

		if initvals.S == nil {
			blas.Copy(initvals.S, s)
		} else {
			is := make([]int, 0)
			is = append(is, matrix.MakeIndexSet(0, dims.At("l")[0], 1)...)
			is = append(is, indq[:len(indq)-1]...)
			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				is = append(is, matrix.MakeIndexSet(ind, ind+m*m, m+1)...)
				ind += m*m
			}
			for _, k := range is {
				s.SetIndex(k, 1.0)
			}
		}
		
		if initvals.Y == nil {
			blas.Copy(initvals.Y, y)
		} else {
			blas.ScalFloat(y, 0.0)
		}

		if initvals.Z == nil {
			blas.Copy(initvals.Z, z)
		} else {
			is := make([]int, 0)
			is = append(is, matrix.MakeIndexSet(0, dims.At("l")[0], 1)...)
			is = append(is, indq[:len(indq)-1]...)
			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				is = append(is, matrix.MakeIndexSet(ind, ind+m*m, m+1)...)
				ind += m*m
			}
			for _, k := range is {
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

	gap = Sdot(s, z, dims, 0)
	fmt.Printf("== pre-loop:\n")
	fmt.Printf("x=\n%v\n", x.ToString("%.17f"))
	fmt.Printf("y=\n%v\n", y.ToString("%.17f"))
	fmt.Printf("s=\n%v\n", s.ToString("%.17f"))
	fmt.Printf("z=\n%v\n", z.ToString("%.17f"))
	for iter := 0; iter < solopts.MaxIter+1; iter++ {
		fmt.Printf("== iterations %d:\n", iter)

        // f0 = (1/2)*x'*P*x + q'*x + r and  rx = P*x + q + A'*y + G'*z.
        blas.Copy(q, rx)
        fP(x, rx, 1.0, 1.0)
        f0 = 0.5 * (blas.DotFloat(x, rx) + blas.DotFloat(x, q))
        fA(y, rx, 1.0, 1.0, la_.OptTrans)
        fG(z, rx, 1.0, 1.0, la_.OptTrans)
        resx = math.Sqrt(blas.DotFloat(rx, rx))
           
        // ry = A*x - b
        blas.Copy(b, ry)
        fA(x, ry, 1.0, -1.0)
        resy = math.Sqrt(blas.DotFloat(ry, ry))

        // rz = s + G*x - h
        blas.Copy(s, rz)
        blas.AxpyFloat(h, rz, -1.0)
        fG(x, rz, 1.0, 1.0)
        resz = Snrm2(rz, dims, 0)
        // Statistics for stopping criteria.

        // pcost = (1/2)*x'*P*x + q'*x 
        // dcost = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h) '
        //       = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h+s) - z'*s
        //       = (1/2)*x'*P*x + q'*x + y'*ry + z'*rz - gap
        pcost = f0
        dcost = f0 + blas.DotFloat(y, ry) + Sdot(z, rz, dims, 0) - gap
        if pcost < 0.0 {
            relgap = gap / -pcost
        } else if dcost > 0.0 {
            relgap = gap / dcost 
        } else {
            relgap = math.NaN()
		}
        pres = math.Max(resy/resy0, resz/resz0)
        dres = resx/resx0 
		
		if pres <= feasTolerance && dres <= feasTolerance &&
			( gap <= absTolerance || (!math.IsNaN(relgap) && relgap <= relTolerance)) ||
			iter == solopts.MaxIter {

			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				Symm(s, m, ind)
				Symm(z, m, ind)
				ind += m*m
			}
			ts,_ = MaxStep(s, dims, 0, nil)
			tz,_ = MaxStep(z, dims, 0, nil)
			if iter == solopts.MaxIter {
				// terminated on max iterations.
				sol.Status = Unknown
				err = errors.New("Terminated (maximum iterations reached)")
				fmt.Printf("Terminated (maximum iterations reached)\n")
				return
			}
			// optimal solution found
			fmt.Print("Optimal solution.")
			err = nil
			sol.X = x
			sol.Y = y
			sol.S = s
			sol.Z = z
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
			W, err = ComputeScaling(s, z, lmbda, dims, 0)
			fmt.Printf("-- initial lmbda=\n%v\n", lmbda.ConvertToString())
		}
		Ssqr(lmbdasq, lmbda, dims, 0)
		fmt.Printf("lmbdasq=\n%v\n", lmbdasq.ConvertToString())

		f3, err = kktsolver(W)
		if err != nil {
			if iter == 0 {
				err = errors.New("Rank(A) < p or Rank([P; A; G]) < n")
				return
			} else {
				ind := dims.Sum("l", "q")
				for _, m := range dims.At("s") {
					Symm(s, m, ind)
					Symm(z, m, ind)
					ind += m*m
				}
				ts,_ = MaxStep(s, dims, 0, nil)
				tz,_ = MaxStep(z, dims, 0, nil)
				// terminated (singular KKT matrix)
				fmt.Printf("Terminated (singular KKT matrix).\n")
				err = errors.New("Terminated (singular KKT matrix).")
				sol.X = x; sol.Y = y; sol.S = s; sol.Z = z
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
			Sinv(s, lmbda, dims, 0)

            // z := z - W'*s 
            //    = bz - W'*(lambda o\ bs)
			blas.Copy(s, ws3)
			Scale(ws3, W, true, false)
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
			blas.AxpyFloat(lmbdasq, ds, -1.0, &la_.IOpt{"n", dims.Sum("l", "q")})
			ind := dims.At("l")[0]
			ds.AddAt(matrix.MakeIndexSet(0, ind, 1), sigma*mu)
			for _, m := range dims.At("q") {
				ds.SetIndex(ind, sigma*mu+ds.GetIndex(ind))
				ind += m
			}
			ind2 := ind
			for _, m := range dims.At("s") {
				blas.AxpyFloat(lmbdasq, ds, -1.0, &la_.IOpt{"n", m}, &la_.IOpt{"incy", m+1},
					&la_.IOpt{"offsetx", ind2}, &la_.IOpt{"offsety", ind})
				ds.AddAt(matrix.MakeIndexSet(ind, ind+m*m, m+1), sigma*mu)
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

			fmt.Printf("== Calling f4 %d\n", i)
			fmt.Printf("dx=\n%v\n", dx.ToString("%.17f"))
			fmt.Printf("ds=\n%v\n", ds.ToString("%.17f"))
			fmt.Printf("dz=\n%v\n", dz.ToString("%.17f"))
			fmt.Printf("== Entering f4 %d\n", i)
			err = f4(dx, dy, dz, ds)
			fmt.Printf("== Return from f4 %d\n", i)
			fmt.Printf("dx=\n%v\n", dx.ToString("%.17f"))
			fmt.Printf("ds=\n%v\n", ds.ToString("%.17f"))
			fmt.Printf("dz=\n%v\n", dz.ToString("%.17f"))
			fmt.Printf("== End from f4 %d\n", i)
			if err != nil {
				if iter == 0 {
					err = errors.New("Rank(A) < p or Rank([P; A; G]) < n")
					return
				} else {
					ind = dims.Sum("l", "q")
					for _, m := range dims.At("s") {
						Symm(s, m, ind)
						Symm(z, m, ind)
						ind += m*m
					}
					ts,_ = MaxStep(s, dims, 0, nil)
					tz,_ = MaxStep(z, dims, 0, nil)
					return
				}
			}

			dsdz := Sdot(ds, dz, dims, 0)
			if correction && i == 0 {
				blas.Copy(ds, ws3)
				Sprod(ws3, dz, dims, 0)
			}

            // Maximum step to boundary.
            // 
            // If i is 1, also compute eigenvalue decomposition of the 's' 
            // blocks in ds, dz.  The eigenvectors Qs, Qz are stored in 
            // dsk, dzk.  The eigenvalues are stored in sigs, sigz. 
			Scale2(lmbda, ds, dims, 0, false)
			Scale2(lmbda, dz, dims, 0, false)
			if i == 0 {
				ts,_ = MaxStep(ds, dims, 0, nil)
				tz,_ = MaxStep(dz, dims, 0, nil)
			} else {
				ts,_ = MaxStep(ds, dims, 0, sigs)
				tz,_ = MaxStep(dz, dims, 0, sigz)
			}
			t := maxvec([]float64{0.0, ts, tz})
			fmt.Printf("== t=%.17f from %v\n", t, []float64{ts, tz})
			var step float64
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
			fmt.Printf("== step=%.17f sigma=%.17f dsdz=%.17f\n", step, sigma, dsdz)

		}

		blas.AxpyFloat(dx, x, step)
		blas.AxpyFloat(dy, y, step)
		fmt.Printf("x=\n%v\n", x.ConvertToString())
		fmt.Printf("y=\n%v\n", y.ConvertToString())
		fmt.Printf("ds=\n%v\n", ds.ConvertToString())
		fmt.Printf("dz=\n%v\n", dz.ConvertToString())
		fmt.Printf("---\n")
        // We will now replace the 'l' and 'q' blocks of ds and dz with 
        // the updated iterates in the current scaling.
        // We also replace the 's' blocks of ds and dz with the factors 
        // Ls, Lz in a factorization Ls*Ls', Lz*Lz' of the updated variables
        // in the current scaling.

        // ds := e + step*ds for nonlinear, 'l' and 'q' blocks.
        // dz := e + step*dz for nonlinear, 'l' and 'q' blocks.
		blas.ScalFloat(ds, step, &la_.IOpt{"n", dims.Sum("l", "q")})
		blas.ScalFloat(dz, step, &la_.IOpt{"n", dims.Sum("l", "q")})
		ind := dims.At("l")[0]
		ds.AddAt(matrix.MakeIndexSet(0, ind, 1), 1.0)
		dz.AddAt(matrix.MakeIndexSet(0, ind, 1), 1.0)
		for _, m := range dims.At("q") {
			ds.SetIndex(ind, 1.0+ds.GetIndex(ind))
			dz.SetIndex(ind, 1.0+dz.GetIndex(ind))
			ind += m
		}
		fmt.Printf("ds=\n%v\n", ds.ConvertToString())
		fmt.Printf("dz=\n%v\n", dz.ConvertToString())

        // ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
        // 
        // This replaces the 'l' and 'q' components of ds and dz with the
        // updated variables in the current scaling.  
        // The 's' components of ds and dz are replaced with 
        // 
        // diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2} 
        // diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2} 
		Scale2(lmbda, ds, dims, 0, true)
		Scale2(lmbda, dz, dims, 0, true)

        // sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
        // sigz := ( e + step*sigz ) ./ lambda for 's' blocks.
        blas.ScalFloat(sigs, step)
        blas.ScalFloat(sigz, step)
        sigs.Add(1.0)
        sigz.Add(1.0)
		sdimsum := dims.Sum("s")
		qdimsum := dims.Sum("l", "q")
		blas.TbsvFloat(lmbda, sigs, &la_.IOpt{"n", sdimsum}, &la_.IOpt{"k", 0},
			&la_.IOpt{"lda", 1}, &la_.IOpt{"offseta", qdimsum})
		blas.TbsvFloat(lmbda, sigz, &la_.IOpt{"n", sdimsum}, &la_.IOpt{"k", 0},
			&la_.IOpt{"lda", 1}, &la_.IOpt{"offseta", qdimsum})
		
		ind2 := qdimsum; ind3 := 0
		sdims := dims.At("s")
		
		for k := 0; k < len(sdims); k++ {
			m := sdims[k]
			for i := 0; i < m; i++ {
				a := math.Sqrt(sigs.GetIndex(ind3+i))
				blas.ScalFloat(ds, a, &la_.IOpt{"offset", ind2+m*i}, &la_.IOpt{"n", m})
				a = math.Sqrt(sigz.GetIndex(ind3+i))
				blas.ScalFloat(dz, a, &la_.IOpt{"offset", ind2+m*i}, &la_.IOpt{"n", m})
			}
			ind2 += m*m
			ind3 += m
		}
		
		//fmt.Printf("pre update_scaling lmbda=\n%v\nds=\n%v\ndz=\n%v\n", lmbda, ds, dz)
		err = UpdateScaling(W, lmbda, ds, dz)
		fmt.Printf("== post update_scaling\nlmbda=\n%v\nds=\n%v\ndz=\n%v\n", lmbda.ConvertToString(), ds.ConvertToString(), dz.ConvertToString())
        // Unscale s, z, tau, kappa (unscaled variables are used only to 
        // compute feasibility residuals).
		ind = dims.Sum("l", "q")
		ind2 = ind
		blas.Copy(lmbda, s, &la_.IOpt{"n", ind})
		for _, m := range dims.At("s") {
			blas.ScalFloat(s, 0.0, &la_.IOpt{"offset", ind2})
			blas.Copy(lmbda, s, &la_.IOpt{"offsetx", ind}, &la_.IOpt{"offsety", ind2},
				&la_.IOpt{"n", m}, &la_.IOpt{"incy", m+1})
			ind += m
			ind2 += m*m
		}
		Scale(s, W, true, false)
		
		ind = dims.Sum("l", "q")
		ind2 = ind
		blas.Copy(lmbda, z, &la_.IOpt{"n", ind})
		for _, m := range dims.At("s") {
			blas.ScalFloat(z, 0.0, &la_.IOpt{"offset", ind2})
			blas.Copy(lmbda, z, &la_.IOpt{"offsetx", ind}, &la_.IOpt{"offsety", ind2},
				&la_.IOpt{"n", m}, &la_.IOpt{"incy", m+1})
			ind += m
			ind2 += m*m
		}
		Scale(z, W, false, true)

		gap = blas.DotFloat(lmbda, lmbda)
		fmt.Printf("== gap = %.17f\n", gap)
	}
	return
}

// Local Variables:
// tab-width: 4
// End:
