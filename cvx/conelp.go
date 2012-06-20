
package cvx

import (
	la_ "github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/matrix"
	"errors"
	"fmt"
	"math"
)

type KKTFunc func(x, y, z *matrix.FloatMatrix) error
type KKTSolver func(*FloatMatrixSet, *matrix.FloatMatrix, *matrix.FloatMatrix)KKTFunc

func KKTNullSolver(W *FloatMatrixSet) KKTFunc {
	nullsolver := func(x, y, z *matrix.FloatMatrix) error {
		return nil
	}
	return nullsolver
}

type SolverMap map[string]KKTSolver

var solvers SolverMap = SolverMap{
	"ldl": nil, "ldl2": nil, "qr": nil, "chol": nil, "chol2": nil}

func checkConeLpDimensions(dims *DimensionSet) error {
	return nil
}

func sgemv(G, x, y *matrix.FloatMatrix, alpha, beta matrix.FScalar, dims *DimensionSet, opts ...la_.Option) error {
	return nil
}

// Set vector indexes from start to end-1 to val.
func AddToFloatVector(vec *matrix.FloatMatrix, start, end int, val float64) {
	if start < 0 {
		start = 0
	}
	if end < 0 {
		end = 0
	}
	if end > vec.NumElements() {
		end = vec.NumElements()
	}
	for k := start; k < end; k++ {
		vec.SetIndex(k, val+vec.GetIndex(k))
	}
}

type StatusCode int
const (
	Optimal = StatusCode(1 + iota)
	PrimalInfeasible
	DualInfeasible
	Unknown
)

type Solution struct {
	Status StatusCode
	X *matrix.FloatMatrix
	S *matrix.FloatMatrix
	Z *matrix.FloatMatrix
	Y *matrix.FloatMatrix
	PrimalObjective float64
	DualObjective float64
	Gap float64
	RelativeGap float64
	PrimalInfeasibility float64
	DualInfeasibility float64
	PrimalSlack float64
	DualSlack float64
	PrimalResidualCert float64
	DualResidualCert float64
	Iterations int
}

type SolverOptions struct {
	AbsTol float64
	RelTol float64
	FeasTol float64
	MaxIter int
	ShowProgress bool
}

// ConeLp(c, G, h, A=nil, b=nil, dims=nil, ...)
func ConeLp(c, G, h, A, b *matrix.FloatMatrix, dims *DimensionSet, solopts *SolverOptions, opts ...la_.Option) (sol *Solution, err error) {

	err = nil
	sol = &Solution{Unknown,
		nil, nil, nil, nil,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0}

	var primalstart *FloatMatrixSet = nil
	var dualstart *FloatMatrixSet = nil
	var kktsolver KKTSolver = nil
	//primalstart = FSetNew("x", "s")
	//dualstart = FSetNew("y", "z")

	// we are missing all sort of checks here ...

	if dims != nil && (len(dims.At("q")) > 0 || len(dims.At("s")) > 0) {
		kktsolver = solvers["qr"]
	} else {
		kktsolver = solvers["chol2"]
	}

	if c == nil || c.Cols() > 1 {
		err = errors.New("'c' must be matrix with 1 column")
		return 
	}
	if h == nil || h.Cols() > 1 {
		err = errors.New("'h' must be matrix with 1 column")
		return 
	}

	if dims == nil {
		dims = DSetNew("l", "q", "s")
		dims.Set("l", []int{h.Rows()})
	}
	if err = checkConeLpDimensions(dims); err != nil {
		return 
	}

	cdim := dims.Sum("l", "q") + dims.SumSquared("s")
	cdim_pckd := dims.Sum("l", "q") + dims.SumPacked("s")
	cdim_diag := dims.Sum("l", "q", "s")

	if h.Rows() != cdim {
		err = errors.New(fmt.Sprintf("'h' must be float matrix of size (%d,1)", cdim))
		return 
	}

	// Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G.
	indq := make([]int, 10, 100)
	indq = append(indq, dims.At("l")[0])
	for _, k := range dims.At("q") {
		indq = append(indq, indq[len(indq)-1]+k)
	}

    // Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G.
	inds := make([]int, 10, 100)
	inds = append(inds, indq[len(indq)-1])
	for _, k := range dims.At("q") {
		inds = append(inds, inds[len(inds)-1]+k*k)
	}

	if G != nil && !G.SizeMatch(cdim, c.Rows()) {
		estr := fmt.Sprintf("'G' must be of size (%d,%d)", cdim, c.Rows())
		err = errors.New(estr)
		return 
	}
	Gf := func(x, y *matrix.FloatMatrix, alpha, beta matrix.FScalar, opts ...la_.Option) error{
		return sgemv(G, x, y, alpha, beta, dims, opts...)
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

	Af := func(x, y *matrix.FloatMatrix, alpha, beta matrix.FScalar, opts ...la_.Option) error {
		return blas.Gemv(A, x, y, alpha, beta, opts...)
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

    // res() evaluates residual in 5x5 block KKT system
    //
    //     [ vx   ]    [ 0         ]   [ 0   A'  G'  c ] [ ux        ]
    //     [ vy   ]    [ 0         ]   [-A   0   0   b ] [ uy        ]
    //     [ vz   ] += [ W'*us     ] - [-G   0   0   h ] [ W^{-1}*uz ]
    //     [ vtau ]    [ dg*ukappa ]   [-c' -b' -h'  0 ] [ utau/dg   ]
    // 
    //           vs += lmbda o (dz + ds) 
    //       vkappa += lmbdg * (dtau + dkappa).
	ws3 := matrix.FloatZeros(cdim, 1)
	wz3 := matrix.FloatZeros(cdim, 1)

	// 
	res := func(ux, uy, uz, utau, us, ukappa, vx, vy, vz, vtau, vs, vkappa *matrix.FloatMatrix, W *FloatMatrixSet, dg matrix.FScalar, lmbda *matrix.FloatMatrix) {

		// vx := vx - A'*uy - G'*W^{-1}*uz - c*utau/dg
        Af(uy, vx, matrix.FScalar(-1.0), matrix.FScalar(1.0), la_.OptTrans)
        blas.Copy(uz, wz3)
        Scale(wz3, W, false, true)
        Gf(wz3, vx, matrix.FScalar(-1.0), matrix.FScalar(1.0), la_.OptTrans)
        blas.Axpy(c, vx, matrix.FScalar(-utau.GetIndex(0)/dg.Float()))

        // vy := vy + A*ux - b*utau/dg
        Af(ux, vy, matrix.FScalar(1.0), matrix.FScalar(1.0))
        blas.Axpy(b, vy, matrix.FScalar(-utau.GetIndex(0)/dg.Float()))
 
        // vz := vz + G*ux - h*utau/dg + W'*us
        Gf(ux, vz, matrix.FScalar(1.0), matrix.FScalar(1.0))
        blas.Axpy(h, vz, matrix.FScalar(-utau.GetIndex(0)/dg.Float()))
        blas.Copy(us, ws3)
        Scale(ws3, W, true, false)
        blas.Axpy(ws3, vz, matrix.FScalar(1.0))

        // vtau := vtau + c'*ux + b'*uy + h'*W^{-1}*uz + dg*ukappa
        var vtauplus float64 = dg.Float()*ukappa.GetIndex(0) + blas.Dot(c,ux).Float() +
			blas.Dot(b,uy).Float() + Sdot(h, wz3, dims, 0) 
		vtau.SetIndex(0, vtau.GetIndex(0)+vtauplus)

        // vs := vs + lmbda o (uz + us)
        blas.Copy(us, ws3)
        blas.Axpy(uz, ws3, matrix.FScalar(1.0))
        Sprod(ws3, lmbda, dims, 0, &la_.SOpt{"diag", "D"})
        blas.Axpy(ws3, vs, matrix.FScalar(1.0))

        // vkappa += vkappa + lmbdag * (utau + ukappa)
		lscale := lmbda.GetIndex(lmbda.NumElements()-1)
		var vkplus float64 = lscale * (utau.GetIndex(0) + ukappa.GetIndex(0))
		vkappa.SetIndex(0, vkappa.GetIndex(0)+vkplus)
	}

	resx0 := math.Max(1.0, math.Sqrt(blas.Dot(c,c).Float()))
	resy0 := math.Max(1.0, math.Sqrt(blas.Dot(b,b).Float()))
	resz0 := math.Max(1.0, Snrm2(h, dims, 0))

	// select initial points

	x := c.Copy()
	blas.Scal(x, matrix.FScalar(0.0))
	y := b.Copy()
	blas.Scal(y, matrix.FScalar(0.0))
	s := matrix.FloatZeros(cdim, 1)
	z := matrix.FloatZeros(cdim, 1)
	dx := c.Copy()
	dy := b.Copy()
	ds := matrix.FloatZeros(cdim, 1)
	dz := matrix.FloatZeros(cdim, 1)
	dkappa := matrix.FloatZeros(1,1)
	dtau := matrix.FloatZeros(1,1)

	var W *FloatMatrixSet
	var f KKTFunc
	if primalstart == nil || dualstart == nil {
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
			W.Append("r", matrix.FloatIdentity(n, n))
			W.Append("rti", matrix.FloatIdentity(n, n))
		}
		f = kktsolver(W, nil, nil)
	}

	if primalstart == nil {
        // minimize    || G * x - h ||^2
        // subject to  A * x = b
        //
        // by solving
        //
        //     [ 0   A'  G' ]   [ x  ]   [ 0 ]
        //     [ A   0   0  ] * [ dy ] = [ b ].
        //     [ G   0  -I  ]   [ -s ]   [ h ]
		blas.Scal(x, matrix.FScalar(0.0))
		blas.Copy(y, dy)
		blas.Copy(h, s)
		err = f(x, dy, s)
		if err != nil {
			return
		}
		blas.Scal(s, matrix.FScalar(-1.0))
	} else {
		blas.Copy(primalstart.At("x")[0], x)
		blas.Copy(primalstart.At("s")[0], s)
	}
		
    // ts = min{ t | s + t*e >= 0 }
    ts := MaxStep(s, dims, 0, nil)
    if ts >= 0 && primalstart != nil {
		err = errors.New("initial s is not positive")
        return 
	}

	if dualstart == nil {
        // minimize   || z ||^2
        // subject to G'*z + A'*y + c = 0
        //
        // by solving
        //
        //     [ 0   A'  G' ] [ dx ]   [ -c ]
        //     [ A   0   0  ] [ y  ] = [  0 ].
        //     [ G   0  -I  ] [ z  ]   [  0 ]
		blas.Copy(c, dx)
		blas.Scal(dx, matrix.FScalar(-1.0))
		blas.Scal(y, matrix.FScalar(0.0))
		if err = f(dx, y, z); err != nil {
			return
		}
	} else {
		if my := dualstart.At("y")[0]; my !=nil {
			blas.Copy(my, y)
		}
		blas.Copy(dualstart.At("z")[0], z)
	}

	// ts = min{ t | z + t*e >= 0 }
    tz := MaxStep(z, dims, 0, nil)
    if tz >= 0 && dualstart != nil {
		err = errors.New("initial z is not positive")
        return 
	}

	nrms := Snrm2(s, dims, 0)
	nrmz := Snrm2(z, dims, 0)

	gap := 0.0
	pcost := 0.0
	dcost := 0.0
	relgap := 0.0

	if primalstart == nil && dualstart == nil {
		gap = Sdot(s, z, dims, 0)
		pcost = blas.Dot(c, x).Float()
		dcost = -blas.Dot(b, y).Float() - Sdot(h, z, dims, 0)
		if pcost < 0.0 {
			relgap = gap / -pcost
		} else if dcost > 0.0 {
			relgap = gap / dcost
		} else {
			relgap = math.NaN()
		}
		if ts <= 0 && tz < 0 &&
			( gap <= solopts.AbsTol || (!math.IsNaN(relgap) && relgap <= solopts.RelTol)) {
			// Constructed initial points happen to be feasible and optimal

			ind := dims.At("l")[0] + dims.Sum("q")
			for _, m := range dims.At("s") {
				Symm(s, m, ind)
				Symm(z, m, ind)
				ind += m*m
			}

			// rx = A'*y + G'*z + c
			rx := c.Copy()
			Af(y, rx, matrix.FScalar(1.0), matrix.FScalar(1.0), la_.OptTrans)
			Gf(z, rx, matrix.FScalar(1.0), matrix.FScalar(1.0), la_.OptTrans)
			resx := math.Sqrt(blas.Dot(rx, rx).Float())
            // ry = b - A*x 
			ry := b.Copy()
			Af(x, ry, matrix.FScalar(-1.0), matrix.FScalar(-1.0))
			resy := math.Sqrt(blas.Dot(ry, ry).Float())
			// rz = s + G*x - h 
			rz := matrix.FloatZeros(cdim, 1)
			Gf(x, rz, matrix.FScalar(1.0), matrix.FScalar(0.0))
			blas.Axpy(s, rz, matrix.FScalar(1.0))
			blas.Axpy(h, rz, matrix.FScalar(-1.0))
			resz := Snrm2(rz, dims, 0)

			pres := math.Max(resy/resy0, resz/resz0)
			dres := resx/resx0
			cx := blas.Dot(c, x).Float()
			by := blas.Dot(b, y).Float()
			hz := Sdot(h, z, dims, 0)

			sol.X = x; sol.Y = y; sol.S = s; sol.Z = z
			sol.Status = Optimal
			sol.Gap = gap; sol.RelativeGap = relgap
			sol.PrimalObjective = cx
			sol.DualObjective = -(by + hz)
			sol.PrimalInfeasibility = pres
			sol.DualInfeasibility = dres
			sol.PrimalSlack = -ts
			sol.DualSlack = -tz

			return 
		}
		
		if ts >= -1e-8 * math.Max(nrms, 1.0) {
            // a = 1.0 + ts  
			a := 1.0 + ts
			addA := func(v float64)float64 {
				return v+a
			}
            // s[:dims['l']] += a
			s.ApplyToIndexes(nil, matrix.MakeIndexSet(0, dims.At("l")[0], 1), addA)
            // s[indq[:-1]] += a
			s.ApplyToIndexes(nil, indq, addA)
            // ind = dims['l'] + sum(dims['q'])
			ind := dims.Sum("l", "q")
            // for m in dims['s']:
            //    s[ind : ind+m*m : m+1] += a
            //    ind += m**2
			for _, m := range dims.At("s") {
				iset := matrix.MakeIndexSet(ind, ind+m*m, m+1)
				s.ApplyToIndexes(nil, iset, addA)
				ind += m*m
			}
		}

		if tz >= -1e-8 * math.Max(nrmz, 1.0) {
            // a = 1.0 + ts  
			a := 1.0 + ts
			addA := func(v float64)float64 {
				return v+a
			}
            // z[:dims['l']] += a
			z.ApplyToIndexes(nil, matrix.MakeIndexSet(0, dims.At("l")[0], 1), addA)
            // z[indq[:-1]] += a
			z.ApplyToIndexes(nil, indq, addA)
            // ind = dims['l'] + sum(dims['q'])
			ind := dims.Sum("l", "q")
            // for m in dims['s']:
            //    z[ind : ind+m*m : m+1] += a
            //    ind += m**2
			for _, m := range dims.At("s") {
				iset := matrix.MakeIndexSet(ind, ind+m*m, m+1)
				z.ApplyToIndexes(nil, iset, addA)
				ind += m*m
			}
		}
	} else if primalstart == nil && dualstart != nil {
		if ts >= -1e-8 * math.Max(nrms, 1.0) {
			a := 1.0 + ts
			addA := func(v float64)float64 {
				return v+a
			}
			s.ApplyToIndexes(nil, matrix.MakeIndexSet(0, dims.At("l")[0], 1), addA)
			s.ApplyToIndexes(nil, indq, addA)
			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				iset := matrix.MakeIndexSet(ind, ind+m*m, m+1)
				s.ApplyToIndexes(nil, iset, addA)
				ind += m*m
			}
		}

	} else if primalstart != nil && dualstart == nil {
		if tz >= -1e-8 * math.Max(nrmz, 1.0) {
			a := 1.0 + ts
			addA := func(v float64)float64 {
				return v+a
			}
			z.ApplyToIndexes(nil, matrix.MakeIndexSet(0, dims.At("l")[0], 1), addA)
			z.ApplyToIndexes(nil, indq, addA)
			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				iset := matrix.MakeIndexSet(ind, ind+m*m, m+1)
				z.ApplyToIndexes(nil, iset, addA)
				ind += m*m
			}
		}
	}
	
	tau := matrix.FScalar(1.0)
	kappa := matrix.FScalar(1.0)

	rx  := c.Copy()
	hrx := c.Copy()
	ry  := b.Copy()
	hry := b.Copy()
	rz  := matrix.FloatZeros(cdim, 1)
	hrz := matrix.FloatZeros(cdim, 1)
	sigs := matrix.FloatZeros(dims.Sum("s"), 1)
	sigz := matrix.FloatZeros(dims.Sum("s"), 1)
	lmbda := matrix.FloatZeros(cdim_diag+1, 1)
	lmbdasq := matrix.FloatZeros(cdim_diag+1, 1)
	
	gap = Sdot(s, z, dims, 0)

	for iter := 0; iter <= solopts.MaxIter; iter++ {
        // hrx = -A'*y - G'*z 
        Af(y, hrx, matrix.FScalar(-1.0), matrix.FScalar(0.0), la_.OptTrans)
        Gf(z, hrx, matrix.FScalar(-1.0), matrix.FScalar(1.0), la_.OptTrans)
        hresx := math.Sqrt( blas.Dot(hrx, hrx).Float() ) 

        // rx = hrx - c*tau 
        //    = -A'*y - G'*z - c*tau
        blas.Copy(hrx, rx)
        blas.Axpy(c, rx, matrix.FScalar(-tau))
        resx := math.Sqrt( blas.Dot(rx, rx).Float() ) / tau.Float()

        // hry = A*x  
        Af(x, hry, matrix.FScalar(1.0), matrix.FScalar(0.0))
        hresy := math.Sqrt( blas.Dot(hry, hry).Float() )

        // ry = hry - b*tau 
        //    = A*x - b*tau
        ry = hry.Copy()
        blas.Axpy(b, ry, matrix.FScalar(-tau))
        resy := math.Sqrt( blas.Dot(ry, ry).Float() ) / tau.Float()

        // hrz = s + G*x  
        Gf(x, hrz, matrix.FScalar(1.0), matrix.FScalar(0.0))
        blas.Axpy(s, hrz, matrix.FScalar(1.0))
        hresz := Snrm2(hrz, dims, 0) 

        // rz = hrz - h*tau 
        //    = s + G*x - h*tau
        blas.Scal(rz, matrix.FScalar(0.0))
        blas.Axpy(hrz, rz, matrix.FScalar(1.0))
        blas.Axpy(h, rz, matrix.FScalar(-tau))
        resz := Snrm2(rz, dims, 0) / tau.Float()

        // rt = kappa + c'*x + b'*y + h'*z 
		cx := blas.Dot(c, x).Float()
		by := blas.Dot(b, y).Float()
		hz := Sdot(h, z, dims, 0)
        rt := kappa.Float() + cx + by + hz 
		if rt == 0.0 {
			// remove this
		}

		pcost = cx/tau.Float()
		dcost = -(by + hz) / tau.Float()
		if pcost < 0.0 {
			relgap = gap / -pcost
		} else if dcost > 0.0 {
			relgap = gap / dcost
		} else {
			relgap = math.NaN()
		}
		pres := math.Max(resy/resy0, resz/resz0)
		dres := resx/resx0
		pinfres := math.NaN()
		if hz + by < 0.0 {
			pinfres = hresx / resx0 / (-hz - by)
		}
		dinfres := math.NaN()
		if cx < 0.0 {
			dinfres = math.Max(hresy/resy0, hresz/resz0) / (-cx)
		}
		if solopts.ShowProgress {
			// show something
		}
		if (pres <= solopts.FeasTol && dres <= solopts.FeasTol &&
			(gap <= solopts.AbsTol || (!math.IsNaN(relgap) && relgap <= solopts.RelTol))) ||
			iter == solopts.MaxIter {
			// done
			blas.Scal(x, matrix.FScalar(1.0/tau))
			blas.Scal(y, matrix.FScalar(1.0/tau))
			blas.Scal(s, matrix.FScalar(1.0/tau))
			blas.Scal(z, matrix.FScalar(1.0/tau))
			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				Symm(s, m, ind)
				Symm(z, m, ind)
				ind += m*m
			}
			ts = MaxStep(s, dims, 0, nil)
			tz = MaxStep(z, dims, 0, nil)
			if iter == solopts.MaxIter {
				// MaxIterations exceeded
				sol.X = x; sol.Y = y; sol.S = s; sol.Z = z
				sol.Status = Unknown
				sol.Gap = gap; sol.RelativeGap = relgap
				sol.PrimalObjective = pcost
				sol.DualObjective = dcost
				sol.PrimalInfeasibility = pres
				sol.DualInfeasibility = dres
				sol.PrimalSlack = -ts
				sol.DualSlack = -tz
				sol.PrimalResidualCert = pinfres
				sol.DualResidualCert = dinfres
				sol.Iterations = iter
				return
			} else {
				// Optimal
				sol.X = x; sol.Y = y; sol.S = s; sol.Z = z
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
		} else if ! math.IsNaN(pinfres) && pinfres <= solopts.FeasTol {
			// Primal Infeasible
			blas.Scal(y, matrix.FScalar(1.0/(-hz - by)))
			blas.Scal(z, matrix.FScalar(1.0/(-hz - by)))
			sol.X = nil; sol.Y = nil; sol.S = nil; sol.Z = nil
			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				Symm(z, m, ind)
				ind += m*m
			}
			tz = MaxStep(z, dims, 0, nil)
			sol.Status = PrimalInfeasible
			sol.Gap = math.NaN()
			sol.RelativeGap = math.NaN()
			sol.PrimalObjective = math.NaN()
			sol.DualObjective = 1.0
			sol.PrimalInfeasibility = math.NaN()
			sol.DualInfeasibility = math.NaN()
			sol.PrimalSlack = math.NaN()
			sol.DualSlack = -tz
			sol.PrimalResidualCert = pinfres
			sol.DualResidualCert = math.NaN()
			sol.Iterations = iter
			return
		} else if ! math.IsNaN(dinfres) && dinfres <= solopts.FeasTol {
			// Dual Infeasible
			blas.Scal(x, matrix.FScalar(1.0/(-cx)))
			blas.Scal(s, matrix.FScalar(1.0/(-cx)))
			sol.X = nil; sol.Y = nil; sol.S = nil; sol.Z = nil
			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				Symm(s, m, ind)
				ind += m*m
			}
			ts = MaxStep(s, dims, 0, nil)
			sol.Status = PrimalInfeasible
			sol.Gap = math.NaN()
			sol.RelativeGap = math.NaN()
			sol.PrimalObjective = 1.0
			sol.DualObjective = math.NaN()
			sol.PrimalInfeasibility = math.NaN()
			sol.DualInfeasibility = math.NaN()
			sol.PrimalSlack = -ts
			sol.DualSlack = math.NaN()
			sol.PrimalResidualCert = math.NaN()
			sol.DualResidualCert = dinfres
			sol.Iterations = iter
			return
		}
		
        // Compute initial scaling W:
        // 
        //     W * z = W^{-T} * s = lambda
        //     dg * tau = 1/dg * kappa = lambdag.
		var dg, dgi matrix.FScalar
		if iter == 0 {
			W, err = ComputeScaling(s, z, lmbda, dims, 0)
			
            //     dg = sqrt( kappa / tau )
            //     dgi = sqrt( tau / kappa )
            //     lambda_g = sqrt( tau * kappa )  
            // 
            // lambda_g is stored in the last position of lmbda.

			dg = matrix.FScalar(math.Sqrt(kappa.Float()/tau.Float()))
			dgi = matrix.FScalar(math.Sqrt(tau.Float()/kappa.Float()))
			lmbda.SetIndex(-1, math.Sqrt(tau.Float()*kappa.Float()))
		}
        // lmbdasq := lmbda o lmbda 
		Ssqr(lmbdasq, lmbda, dims, 0)
		lmbdasq.SetIndex(-1, lmbda.GetIndex(-1))

        // f3(x, y, z) solves    
        //
        //     [ 0  A'  G'   ] [ ux        ]   [ bx ]
        //     [ A  0   0    ] [ uy        ] = [ by ].
        //     [ G  0  -W'*W ] [ W^{-1}*uz ]   [ bz ]
        //
        // On entry, x, y, z contain bx, by, bz.
        // On exit, they contain ux, uy, uz.
        //
        // Also solve
        //
        //     [ 0   A'  G'    ] [ x1        ]          [ c ]
        //     [-A   0   0     ]*[ y1        ] = -dgi * [ b ].
        //     [-G   0   W'*W  ] [ W^{-1}*z1 ]          [ h ]

		f3 := kktsolver(W, nil, nil)
		var x1, y1, z1 *matrix.FloatMatrix
		if iter == 0 {
			x1 = c.Copy()
			y1 = b.Copy()
			z1 = matrix.FloatZeros(cdim, 1)
		}
		blas.Copy(c, x1)
		blas.Scal(x1, matrix.FScalar(-1.0))
		blas.Copy(b, y1)
		blas.Copy(h, z1)
		err = f3(x1, y1, z1)
		blas.Scal(x1, dgi)
		blas.Scal(y1, dgi)
		blas.Scal(z1, dgi)

		if err != nil {
			if iter == 0 && primalstart != nil && dualstart != nil {
				err = errors.New("Rank(A) < p or Rank([G; A]) < n")
				return
			} else {
				blas.Scal(x, tau.Inv())
				blas.Scal(y, tau.Inv())
				blas.Scal(s, tau.Inv())
				blas.Scal(z, tau.Inv())
				ind := dims.Sum("l", "q")
				for _, m := range dims.At("s") {
					Symm(s, m, ind)
					Symm(z, m, ind)
					ind += m*m
				}
				ts = MaxStep(s, dims, 0, nil)
				tz = MaxStep(z, dims, 0, nil)
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

		// f6_no_ir(x, y, z, tau, s, kappa) solves
		//
		//    [ 0         ]   [  0   A'  G'  c ] [ ux        ]    [ bx   ]
		//    [ 0         ]   [ -A   0   0   b ] [ uy        ]    [ by   ]
		//    [ W'*us     ] - [ -G   0   0   h ] [ W^{-1}*uz ] = -[ bz   ]
		//    [ dg*ukappa ]   [ -c' -b' -h'  0 ] [ utau/dg   ]    [ btau ]
		//
		//    lmbda o (uz + us) = -bs
		//    lmbdag * (utau + ukappa) = -bkappa.
		//
		// On entry, x, y, z, tau, s, kappa contain bx, by, bz, btau, 
		// bkappa.  On exit, they contain ux, uy, uz, utau, ukappa.
		f6_no_ir := func(x, y, z *matrix.FloatMatrix, tau, s, kappa *matrix.FloatMatrix) (err error) {
            /* Solve 
            
                 [  0   A'  G'    0   ] [ ux        ]   
                 [ -A   0   0     b   ] [ uy        ]  
                 [ -G   0   W'*W  h   ] [ W^{-1}*uz ] 
                 [ -c' -b' -h'    k/t ] [ utau/dg   ]
            
                       [ bx                    ]
                       [ by                    ]
                     = [ bz - W'*(lmbda o\ bs) ]
                       [ btau - bkappa/tau     ]
            
                 us = -lmbda o\ bs - uz
                 ukappa = -bkappa/lmbdag - utau.

             First solve 
            
                 [ 0  A' G'   ] [ ux        ]   [  bx                    ]
                 [ A  0  0    ] [ uy        ] = [ -by                    ]
                 [ G  0 -W'*W ] [ W^{-1}*uz ]   [ -bz + W'*(lmbda o\ bs) ]
			 */
			err = nil
			minusOne := matrix.Fscalar(-1.0)
			one := matrix.Fscalar(1.0)
            // y := -y = -by
			blas.Scal(y, minusOne)

            // s := -lmbda o\ s = -lmbda o\ bs
			err = Sinv(s, lmbda, dims)
			blas.Scal(s, minusOne)

            // z := -(z + W'*s) = -bz + W'*(lambda o\ bs)
			blas.Copy(s, ws3)
			err = Scale(ws3, W, true, false)
			blas.Axpy(ws3, z, one)
			blas.Scal(z, minusOne)

			err = f3(x, y, z)
			/*
             Combine with solution of 
            
                 [ 0   A'  G'    ] [ x1         ]          [ c ]
                 [-A   0   0     ] [ y1         ] = -dgi * [ b ]
                 [-G   0   W'*W  ] [ W^{-1}*dzl ]          [ h ]
             
             to satisfy
            
                 -c'*x - b'*y - h'*W^{-1}*z + dg*tau = btau - bkappa/tau.
			 */

            // kappa[0] := -kappa[0] / lmbd[-1] = -bkappa / lmbdag
			v := -kappa.GetIndex(0) / lmbda.GetIndex(-1)
			kappa.SetIndex(0, matrix.FScalar(v))

            // tau[0] = tau[0] + kappa[0] / dgi = btau[0] - bkappa / tau
			v = tau.GetIndex(0) + kappa.GetIndex(0)/dgi.Float()
			tau.SetIndex(0,  matrix.FScalar(v))

            //tau[0] = dgi * ( tau[0] + xdot(c,x) + ydot(b,y) + 
            //    misc.sdot(th, z, dims) ) / (1.0 + misc.sdot(z1, z1, dims))
			v = tau.GetIndex(0) + blas.Dot(c, x).Float() + blas.Dot(b, y).Float()
			v *= dgi.Float()
			v2 := Sdot(th, z, dims)/(1.0 + Sdot(z1, z1, dims))
			tau.SetIndex(0, v/v2)

			blas.Axpy(x1, x, matrix.FScalar(v/v2))
			blas.Axpy(y1, y, matrix.FScalar(v/v2))
			blas.Axpy(z1, z, matrix.FScalar(v/v2))

			blas.Axpy(z, s, minusOne)
			kappa.SetIndex(0, kappa.GetIndex(0)-tau.GetIndex(0))

			return

		}

        // f6(x, y, z, tau, s, kappa) solves the same system as f6_no_ir, 
        // but applies iterative refinement.
		var wx, wy, ws, wz, wtau, wkappa *matrix.FloatMatrix
		var wx2, wy2, ws2, wz2, wtau2, wkappa2 *matrix.FloatMatrix

		if iter == 0 {
			if refinement > 0 || solopts.Debug {
				wx = c.Copy()
				wy = b.Copy()
				wz = matrix.FloatZeros(cdim, 1)
				ws = matrix.FloatZeros(cdim, 1)
				wtau = matrix.FloatValue()
				wkappa = matrix.FloatValue()
			} else if refinement > 0 {
				wx2 = c.Copy()
				wy2 = b.Copy()
				wz2 = matrix.FloatZeros(cdim, 1)
				ws2 = matrix.FloatZeros(cdim, 1)
				wtau2 = matrix.FloatValue()
				wkappa2 = matrix.FloatValue()
			}
		}

		f6 := func(x, y, z, tau, s, kappa *matrix.FloatMatrix) (err error) {
			err =  nil
			if refinement > 0 or solopts.Debug {
				blas.Copy(x, wx)
				blas.Copy(y, wy)
				blas.Copy(z, wz)
				blas.Copy(s, ws)
				wtau.SetIndex(0, tau.FloatValue())
				wkappa.SetIndex(0, kappa.FloatValue())
			}
			err = f6_no_ir(x, y, z, tau, s, kappa)
			for i := 0; i < refinement; i++ {
				blas.Copy(wx, wx2)
				blas.Copy(wy, wy2)
				blas.Copy(wz, wz2)
				blas.Copy(ws, ws2)
				wtau2.SetIndex(0, wtau.FloatValue())
				wkappa2.SetIndex(0, wkappa.FloatValue())
				err = res(x, y, z, tau, s, kappa, wx2, wy2, wz2, wtau2,, ws2, wkappa2, W, dg, lmbda)
				err = f6_no_ir(wx2, wy2, wz2, wtau2, ws2, wkappa2)
				blas.Copy(wx2, x)
				blas.Copy(wy2, y)
				blas.Copy(wz2, z)
				blas.Copy(ws2, s)
				v := tau.FloatValue() + wtau2.FloatValue()
				tau.SetIndex(0, v)
				v = kappa.FloatValue() + wkappa2.FloatValue()
				kappa.SetIndex(0, v)
			}
			if solopts.Debug {
				res(x, y, z, tau, s, kappa, wx, wy, wz, wtau, ws, wkappa, W, dg, lmbda)
				fmt.Printf("KKT residuals\n")
			}
		}

        mu := math.Pow(blas.Nrm2(lmbda).Float(),2) / (1 + cdim_diag) 
        sigma := 0.0

		for i := 0; i < 2; i++ {
			var wkappa3 float64

			/*
             Solve
            
                 [ 0         ]   [  0   A'  G'  c ] [ dx        ]
                 [ 0         ]   [ -A   0   0   b ] [ dy        ]
                 [ W'*ds     ] - [ -G   0   0   h ] [ W^{-1}*dz ]
                 [ dg*dkappa ]   [ -c' -b' -h'  0 ] [ dtau/dg   ]
            
                                   [ rx   ]
                                   [ ry   ]
                     = - (1-sigma) [ rz   ]
                                   [ rtau ]
            
                 lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e
                 lmbdag * (dtau + dkappa) = - kappa * tau + sigma*mu
            
             ds = -lmbdasq if i is 0
                = -lmbdasq - dsa o dza + sigma*mu*e if i is 1
             dkappa = -lambdasq[-1] if i is 0 
                    = -lambdasq[-1] - dkappaa*dtaua + sigma*mu if i is 1.
			 */
			ind := dism.Sum("l", "q")
			blas.Copy(lmbdasq, ds, &la_.IOpt{"n", ind})
			ind2 := ind
			blas.Scal(ds, matrix.FScalar(0,0), &la_.IOpt{"offset", ind})
			for _, m := range dims.At("s") {
				blas.Copy(lmbdasq, ds, &la_.IOpt{"n", m}, &la_.IOpt{"offsetx", ind2},
					&la_.IOpt{"offsety", ind}, &la_.IOpt{"incy", m+1})
				ind += m*m
				ind2 += m
			}
			// dkappa[0] = lmbdasq[-1]
			dkappa.SetIndex(0, lmbdasq.GetIndex(-1))
			if i == 1 {
				blas.Axpy(ws3, ds, matrix.FScalar(1.0))
				sigmaMu := func(a float64)float64 {
					return a - sigma*mu
				}
				// ds[:dims['l']] -= sigma*mu
				ds.ApplyToIndexes(nil, MakeIndexSet(0, dims.At("l")[0], 1), sigmaMu)
				// ds[indq[:-1]] -= sigma*mu  !! WHAT IS THIS !!
				ds.ApplyToIndexes(nil, indq, sigmaMu)
				for _, m := range dims.At("s") {
					// ds[ind : ind+m*m : m+1] -= sigma*mu
					ds.ApplyToIndexes(nil, MakeIndexSet(ind, ind+m*m, m+1), sigmaMu)
					ind += m*m
				}
				v := dkappa.FloatValue() + wkappa3 - sigma*mu
				dkappa.SetIndex(0, v)
			}
            // (dx, dy, dz, dtau) = (1-sigma)*(rx, ry, rz, rt)
			blas.Copy(rx, dx)
			blas.Scal(dx, matrix.FScalar(1.0-sigma))
			blas.Copy(ry, dy)
			blas.Scal(dy, matrix.FScalar(1.0-sigma))
			blas.Copy(rz, dz)
			blas.Scal(dz, matrix.FScalar(1.0-sigma))
            // dtau[0] = (1.0 - sigma) * rt 
			dtau.SetIndex(0, (1.0-sigma)*rt)

			err = f6(dx, dy, dz, dtau, ds, dkappa)

			// Save ds o dz and dkappa * dtau for Mehrotra correction
			if i == 0 {
				blas.Copy(ds, ws3)
				Sprod(ws3, dz, dims, 0)
				wkappa3 = dtau.FloatValue() * dkappa.FloatValue()
			}
			/*
             Maximum step to boundary.
            
             If i is 1, also compute eigenvalue decomposition of the 's' 
             blocks in ds, dz.  The eigenvectors Qs, Qz are stored in 
             dsk, dzk.  The eigenvalues are stored in sigs, sigz. 
			 */
			var ts, tz float64
			Scale2(lmbda, ds, dims)
			Scale2(lmbda, dz, dims)
			if i == 0 {
				ts = MaxStep(ds, dims, 0, nil)
				tz = MaxStep(dz, dims, 0, nil)
			} else {
				ts = MaxStep(ds, dims, 0, sigs)
				tz = MaxStep(dz, dims, 0, sigz)
			}
			tt := -dtau.FloatValue() / lmbda.GetIndex(-1)
			tk := -dkappa.FloatValue() / lmbda.GetIndex(-1)
			t := maxvec([]float64{0.0, ts, tz, tt, tk})
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
				sigma = math.Pow((1.0 - step), EXPON)
			}
		}

		// Update x, y
		blas.Axpy(dx, x, matrix.FScalar(step))
		blas.Axpy(dy, y, matrix.FScalar(step))
		/*
          Replace 'l' and 'q' blocks of ds and dz with the updated 
          variables in the current scaling.
          Replace 's' blocks of ds and dz with the factors Ls, Lz in a 
          factorization Ls*Ls', Lz*Lz' of the updated variables in the 
          current scaling.

          ds := e + step*ds for 'l' and 'q' blocks.
          dz := e + step*dz for 'l' and 'q' blocks.
		 */
		blas.Scal(ds, step, &la_.IOpt{"n", dims.Sum("l", "q")})
		blas.Scal(dz, step, &la_.IOpt{"n", dims.Sum("l", "q")})

		addOne := func(v float64)float64 { return v+1.0 }
		ds.ApplyToIndexes(nil, matrix.MakeIndexSet(0, dims.At("l")[0], 1), addOne)
		dz.ApplyToIndexes(nil, matrix.MakeIndexSet(0, dims.At("l")[0], 1), addOne)
		ds.ApplyToIndexes(nil, indq, addOne)
		dz.ApplyToIndexes(nil, indq, addOne)

		/*
          ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
         
          This replaces the 'l' and 'q' components of ds and dz with the
          updated variables in the current scaling.  
          The 's' components of ds and dz are replaced with 
         
              diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2} 
              diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2} 
		 */
		Scale2(lmbda, ds, dims, true)
		Scale2(lmbda, dz, dims, true)

		// !! EMPTY REF !!
		if dg.Float() > 0.0 {
		}
	}
	// NEVER exits here.
	// this to satisfy the compiler
	fmt.Printf("cdim=%d, cdim_pckd=%d, cdim_diag=%d\n", cdim, cdim_pckd, cdim_diag)
	if Gf == nil || Af == nil || kktsolver == nil {
		fmt.Printf("kktsolver or Gf or Af is nil\n")
	}
	if res == nil || ds == nil || dz == nil || dkappa == nil || dtau == nil {
		fmt.Printf("res || ds || dzis nil\n")
	}
	if lmbda == nil || lmbdasq == nil {
		fmt.Printf("lmbda || lmbdasq  nil\n", )
	}

	return 
}



// Local Variables:
// tab-width: 4
// End:
