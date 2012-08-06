
package cvx

import (
	la_ "github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/matrix"
	"errors"
	"fmt"
	"math"
)

// KKTFactory creates solver factor
type KKTFactory func(*matrix.FloatMatrix, *DimensionSet, *matrix.FloatMatrix, int) (KKT)

// KTTFunc solves
type KKTFunc func(x, y, z *matrix.FloatMatrix) error

// KKTFactor produces solver function
type KKTFactor func(*FloatMatrixSet, *matrix.FloatMatrix, *matrix.FloatMatrix)(KKTFunc, error)

// KKTSolver creates problem spesific factor
type KKTSolver func(*matrix.FloatMatrix, *DimensionSet, *matrix.FloatMatrix, int) (KKTFactor, error)

func KKTNullFactor(W *FloatMatrixSet, H, Df *matrix.FloatMatrix) (KKTFunc, error) {
	nullsolver := func(x, y, z *matrix.FloatMatrix) error {
		return nil
	}
	return nullsolver, nil
}

func KKTNullSolver(G *matrix.FloatMatrix, dims *DimensionSet, A *matrix.FloatMatrix) (KKTFactor, error) {
	return KKTNullFactor, nil
}

type SolverMap map[string]KKTSolver
type SolverFactoryMap map[string]KKTFactory

var solvers SolverMap = SolverMap{
	"ldl": KktLdl,
	"ldl2": KktLdl,
	"qr": KktLdl,
	"chol": KktLdl,
	"chol2": KktLdl}

var factories SolverFactoryMap = SolverFactoryMap{
	"ldl": CreateLdlSolver,
	"ldl2": CreateLdlSolver,
	"qr": CreateLdlSolver,
	"chol": CreateLdlSolver,
	"chol2": CreateLdlSolver}

func checkConeLpDimensions(dims *DimensionSet) error {
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
	Debug bool
	Refinement int
	KKTSolverName string
}

type f6Closure struct {
	wx, wy, ws, wz *matrix.FloatMatrix
	wx2, wy2, ws2, wz2 *matrix.FloatMatrix
	// these are singleton matrices
	wtau, wkappa, wtau2, wkappa2 *matrix.FloatMatrix
}

const (
	MAXITERS = 100
	ABSTOL = 1e-7
	RELTOL = 1e-6
	FEASTOL = 1e-7
)

// ConeLp(c, G, h, A=nil, b=nil, dims=nil, ...)
func ConeLp(c, G, h, A, b *matrix.FloatMatrix, dims *DimensionSet, solopts *SolverOptions, opts ...la_.Option) (sol *Solution, err error) {

	err = nil
	EXPON := 3
	STEP := 0.99

	sol = &Solution{Unknown,
		nil, nil, nil, nil,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 0}

	var primalstart *FloatMatrixSet = nil
	var dualstart *FloatMatrixSet = nil
	var refinement int

	if solopts.Refinement > 0 {
		refinement = solopts.Refinement
	} else {
		refinement = 0
		if len(dims.At("q")) > 0 || len(dims.At("s")) > 0 {
			refinement = 1
		}
	}
	

	//primalstart = FSetNew("x", "s")
	//dualstart = FSetNew("y", "z")

	// we are missing all sort of checks here ...

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

	fmt.Printf("cdim: %d, cdim_pckd: %d, cdim_diag: %d\n", cdim, cdim_pckd, cdim_diag)

	if h.Rows() != cdim {
		err = errors.New(fmt.Sprintf("'h' must be float matrix of size (%d,1)", cdim))
		return 
	}

	// Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G.
	indq := make([]int, 0, 100)
	indq = append(indq, dims.At("l")[0])
	for _, k := range dims.At("q") {
		indq = append(indq, indq[len(indq)-1]+k)
	}
	fmt.Printf("** indq = %v\n", indq)
    // Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G.
	inds := make([]int, 0, 100)
	inds = append(inds, indq[len(indq)-1])
	for _, k := range dims.At("s") {
		inds = append(inds, inds[len(inds)-1]+k*k)
	}
	fmt.Printf("** inds = %v\n", inds)

	if G != nil && !G.SizeMatch(cdim, c.Rows()) {
		estr := fmt.Sprintf("'G' must be of size (%d,%d)", cdim, c.Rows())
		err = errors.New(estr)
		return 
	}
	Gf := func(x, y *matrix.FloatMatrix, alpha, beta float64, opts ...la_.Option) error{
		return Sgemv(G, x, y, alpha, beta, dims, opts...)
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

	Af := func(x, y *matrix.FloatMatrix, alpha, beta float64, opts ...la_.Option) error {
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
	var factor KKTFactor
	var kktsolver KKTFactor = nil
	if kktfunc, ok := solvers[solvername]; ok {
		// kkt function returns us problem spesific factor function.
		factor, err = kktfunc(G, dims, A, 0)
		// solver is 
		kktsolver = func(W *FloatMatrixSet, H, Df *matrix.FloatMatrix) (KKTFunc, error) {
			return factor(W, H, Df)
		}
	} else {
		err = errors.New(fmt.Sprintf("solver '%s' not known", solvername))
		return
	}
	/*
	var kktsolver KKT

	if kkt, ok := factories[solvername]; ok {
		if b.Rows() > c.Rows() || b.Rows() + cdim_pckd < c.Rows() {
			err = errors.New("Rank(A) < p or Rank[G; A] < n")
			return
		}
		if kkt == nil {
			err = errors.New(fmt.Sprintf("solver '%s' not yet implemented", solvername))
			return
		}
		kktsolver = kkt(G, dims, A, 0)
	} else {
		err = errors.New(fmt.Sprintf("solver '%s' not known", solvername))
		return
	}
	 */
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
	res := func(ux, uy, uz, utau , us, ukappa, vx, vy, vz, vtau, vs, vkappa *matrix.FloatMatrix, W *FloatMatrixSet, dg float64, lmbda *matrix.FloatMatrix) (err error) {

		err = nil
		fmt.Printf("** start res ...\n")
		// vx := vx - A'*uy - G'*W^{-1}*uz - c*utau/dg
		Af(uy, vx, -1.0, 1.0, la_.OptTrans)
		blas.Copy(uz, wz3)
		Scale(wz3, W, false, true)
		//fmt.Printf("post-scale wz3=\n%v\n", wz3)
		Gf(wz3, vx, -1.0, 1.0, la_.OptTrans)
		blas.AxpyFloat(c, vx, -utau.Float()/dg)

		// vy := vy + A*ux - b*utau/dg
		Af(ux, vy, 1.0, 1.0)
		blas.AxpyFloat(b, vy, -utau.Float()/dg)

		// vz := vz + G*ux - h*utau/dg + W'*us
		Gf(ux, vz, 1.0, 1.0)
		blas.AxpyFloat(h, vz, -utau.Float()/dg)
		fmt.Printf("post-axpy vz=\n%v\n", vz)
		blas.Copy(us, ws3)
		Scale(ws3, W, true, false)
		fmt.Printf("post-scale ws3=\n%v\n", ws3)
		fmt.Printf("pre-scale-apxy vz=\n%v\n", vz)
		fmt.Printf("9: %.16f + %.16f\n", ws3.GetIndex(9), vz.GetIndex(9))
		blas.AxpyFloat(ws3, vz, 1.0)
		fmt.Printf("post-scale-apxy vz=\n%v\n", vz)
		
		// vtau := vtau + c'*ux + b'*uy + h'*W^{-1}*uz + dg*ukappa
		var vtauplus float64 = dg*ukappa.Float() + blas.DotFloat(c, ux) +
			blas.DotFloat(b, uy) + Sdot(h, wz3, dims, 0) 
		vtau.SetValue(vtau.Float()+vtauplus)

		// vs := vs + lmbda o (uz + us)
		blas.Copy(us, ws3)
		blas.AxpyFloat(uz, ws3, 1.0)
		Sprod(ws3, lmbda, dims, 0, &la_.SOpt{"diag", "D"})
		fmt.Printf("post-sprod ws3=\n%v\n", ws3)
		blas.AxpyFloat(ws3, vs, 1.0)
		fmt.Printf("post-sprod-apxy vs=\n%v\n", vs)

		// vkappa += vkappa + lmbdag * (utau + ukappa)
		lscale := lmbda.GetIndex(lmbda.NumElements()-1)
		var vkplus float64 = lscale * (utau.Float() + ukappa.Float())
		vkappa.SetValue(vkappa.Float()+vkplus)
		fmt.Printf("-- res result:\nvz=\n%v\nvs=\n%v\n", vz, vs)
		fmt.Printf("** end res ...\n")
		return 
	}

	resx0 := math.Max(1.0, math.Sqrt(blas.DotFloat(c,c)))
	resy0 := math.Max(1.0, math.Sqrt(blas.DotFloat(b,b)))
	resz0 := math.Max(1.0, Snrm2(h, dims, 0))

	// select initial points

	fmt.Printf("** initial resx0=%.4f, resy0=%.4f, resz0=%.4f \n", resx0, resy0, resz0)

	x := c.Copy()
	blas.ScalFloat(x, 0.0)
	y := b.Copy()
	blas.ScalFloat(y, 0.0)
	s := matrix.FloatZeros(cdim, 1)
	z := matrix.FloatZeros(cdim, 1)
	dx := c.Copy()
	dy := b.Copy()
	fmt.Printf("x = %d, dx = %d \n", x.NumElements(), dx.NumElements())
	fmt.Printf("y = %d, dy = %d \n", y.NumElements(), dy.NumElements())
	ds := matrix.FloatZeros(cdim, 1)
	dz := matrix.FloatZeros(cdim, 1)
	// these are singleton matrix
	dkappa := matrix.FloatValue(0.0)
	dtau := matrix.FloatValue(0.0)

	fmt.Printf("initial point set ...\n")

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
		dd := dims.At("l")[0]
		mat := matrix.FloatOnes(dd, 1)
		W.Set("d", mat)
		mat = matrix.FloatOnes(dd, 1)
		W.Set("di", mat)
		dq := len(dims.At("q"))
		W.Set("beta", matrix.FloatOnes(dq, 1))

		for _, n := range dims.At("q")  {
			vm := matrix.FloatZeros(n, 1)
			vm.SetIndex(0, 1.0)
			W.Append("v", vm)
		}
		for _, n := range dims.At("s") {
			W.Append("r", matrix.FloatIdentity(n, n))
			W.Append("rti", matrix.FloatIdentity(n, n))
		}
		f, err = kktsolver(W, nil, nil)
		//_, err = kktsolver.Factor(W, nil, nil)
		if err != nil {
			fmt.Printf("kktsolver error: %s\n", err)
			return
		}
		//fmt.Printf("** empty W:\n")
		//W.Print()
		//fmt.Printf("** end of empty W:\n")
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
		blas.ScalFloat(x, 0.0)
		blas.CopyFloat(y, dy)
		blas.CopyFloat(h, s)
		err = f(x, dy, s)
		//err = kktsolver.Solve(x, dy, s)
		if err != nil {
			fmt.Printf("f(x,dy,s): %s\n", err)
			return
		}
		blas.ScalFloat(s, -1.0)
		//fmt.Printf("** initial s:\n%v\n", s)
	} else {
		blas.Copy(primalstart.At("x")[0], x)
		blas.Copy(primalstart.At("s")[0], s)
	}

	// ts = min{ t | s + t*e >= 0 }
	ts := MaxStep(s, dims, 0, nil)
	fmt.Printf("** initial ts:\n%v\n", ts)
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
		blas.ScalFloat(dx, -1.0)
		blas.ScalFloat(y, 0.0)
		err = f(dx, y, z)
		//err = kktsolver.Solve(dx, y, z)
		if err != nil {
			fmt.Printf("f(dx,y,z): %s\n", err)
			return
		}
		//fmt.Printf("** initial z:\n%v\n", z)
	} else {
		if my := dualstart.At("y")[0]; my !=nil {
			blas.Copy(my, y)
		}
		blas.Copy(dualstart.At("z")[0], z)
	}

	// ts = min{ t | z + t*e >= 0 }
	tz := MaxStep(z, dims, 0, nil)
	fmt.Printf("** initial tz:\n%v\n", tz)
	if tz >= 0 && dualstart != nil {
		err = errors.New("initial z is not positive")
		return 
	}

	nrms := Snrm2(s, dims, 0)
	nrmz := Snrm2(z, dims, 0)
	fmt.Printf("** nrms=%.4f, nrmz=%.4f\n", nrms, nrmz)

	gap := 0.0
	pcost := 0.0
	dcost := 0.0
	relgap := 0.0

	if primalstart == nil && dualstart == nil {
		gap = Sdot(s, z, dims, 0)
		pcost = blas.DotFloat(c, x)
		dcost = -blas.DotFloat(b, y) - Sdot(h, z, dims, 0)
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
			Af(y, rx, 1.0, 1.0, la_.OptTrans)
			Gf(z, rx, 1.0, 1.0, la_.OptTrans)
			resx := math.Sqrt(blas.Dot(rx, rx).Float())
			// ry = b - A*x 
			ry := b.Copy()
			Af(x, ry, -1.0, -1.0)
			resy := math.Sqrt(blas.Dot(ry, ry).Float())
			// rz = s + G*x - h 
			rz := matrix.FloatZeros(cdim, 1)
			Gf(x, rz, 1.0, 0.0)
			blas.AxpyFloat(s, rz, 1.0)
			blas.AxpyFloat(h, rz, -1.0)
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
			//fmt.Printf("** scaling s ...\n")
			// a = 1.0 + ts  
			a := 1.0 + ts
			addA := func(v float64)float64 {
				return v+a
			}
			// s[:dims['l']] += a
			is := matrix.MakeIndexSet(0, dims.At("l")[0], 1)
			fmt.Printf("** scaling indexes %v\n", is)
			s.ApplyToIndexes(nil, matrix.MakeIndexSet(0, dims.At("l")[0], 1), addA)
			// s[indq[:-1]] += a
			fmt.Printf("** scaling indexes %v\n", indq[:len(indq)-1])
			s.ApplyToIndexes(nil, indq[:len(indq)-1], addA)
			// ind = dims['l'] + sum(dims['q'])
			ind := dims.Sum("l", "q")
			// for m in dims['s']:
			//    s[ind : ind+m*m : m+1] += a
			//    ind += m**2
			for _, m := range dims.At("s") {
				iset := matrix.MakeIndexSet(ind, ind+m*m, m+1)
				fmt.Printf("** scaling [%d;%d:%d] %v\n", ind, ind+m*m, m+1, iset)
				s.ApplyToIndexes(nil, iset, addA)
				ind += m*m
			}
		}

		if tz >= -1e-8 * math.Max(nrmz, 1.0) {
			fmt.Printf("** scaling z ...\n")
			// a = 1.0 + ts  
			a := 1.0 + tz
			addA := func(v float64)float64 {
				return v+a
			}
			// z[:dims['l']] += a
			z.ApplyToIndexes(nil, matrix.MakeIndexSet(0, dims.At("l")[0], 1), addA)
			// z[indq[:-1]] += a
			z.ApplyToIndexes(nil, indq[:len(indq)-1], addA)
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
			s.ApplyToIndexes(nil, indq[:len(indq)-1], addA)
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
			z.ApplyToIndexes(nil, indq[:len(indq)-1], addA)
			ind := dims.Sum("l", "q")
			for _, m := range dims.At("s") {
				iset := matrix.MakeIndexSet(ind, ind+m*m, m+1)
				z.ApplyToIndexes(nil, iset, addA)
				ind += m*m
			}
		}
	}

	tau := matrix.FloatValue(1.0)
	kappa := matrix.FloatValue(1.0)

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

	//fmt.Printf("pre-gap s=\n%v\npre-gap z=\n%v\n", s, z)
	gap = Sdot(s, z, dims, 0)
	fmt.Printf("** iterate %d times [gap=%.4f]... \n", solopts.MaxIter, gap)

	if solopts.ShowProgress {
		// show headers of something
        fmt.Printf("% 10s% 12s% 10s% 8s% 7s\n", "pcost", "dcost", "gap", "pres", "dres")
	}

	var x1, y1, z1 *matrix.FloatMatrix
	var dg, dgi float64
	var th *matrix.FloatMatrix
	var WS f6Closure
	var f3 KKTFunc

	for iter := 0; iter < solopts.MaxIter; iter++ {
		// hrx = -A'*y - G'*z 
		fmt.Printf("^^^^^^^^^ ITERATIOn %d ^^^^^^^^^^\n", iter)
		Af(y, hrx, -1.0, 0.0, la_.OptTrans)
		Gf(z, hrx, -1.0, 1.0, la_.OptTrans)
		hresx := math.Sqrt( blas.DotFloat(hrx, hrx) ) 

		// rx = hrx - c*tau 
		//    = -A'*y - G'*z - c*tau
		//fmt.Printf("hrx=\n%v\n", hrx)
		blas.Copy(hrx, rx)
		//fmt.Printf("rx=\n%v\n", rx)
		err = blas.AxpyFloat(c, rx, -tau.Float())
		//fmt.Printf("axpy err=%v\n", err)
		resx := math.Sqrt( blas.DotFloat(rx, rx) ) / tau.Float()

		// hry = A*x  
		Af(x, hry, 1.0, 0.0)
		hresy := math.Sqrt( blas.DotFloat(hry, hry) )

		// ry = hry - b*tau 
		//    = A*x - b*tau
		blas.Copy(hry, ry)
		blas.AxpyFloat(b, ry, -tau.Float())
		resy := math.Sqrt( blas.DotFloat(ry, ry) ) / tau.Float()

		// hrz = s + G*x  
		Gf(x, hrz, 1.0, 0.0)
		blas.AxpyFloat(s, hrz, 1.0)
		hresz := Snrm2(hrz, dims, 0) 

		// rz = hrz - h*tau 
		//    = s + G*x - h*tau
		blas.ScalFloat(rz, 0.0)
		blas.AxpyFloat(hrz, rz, 1.0)
		blas.AxpyFloat(h, rz, -tau.Float())
		resz := Snrm2(rz, dims, 0) / tau.Float()

		// rt = kappa + c'*x + b'*y + h'*z '
		cx := blas.DotFloat(c, x)
		by := blas.DotFloat(b, y)
		hz := Sdot(h, z, dims, 0)
		rt := kappa.Float() + cx + by + hz 

		// Statistics for stopping criteria
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
		//fmt.Printf("-- tau=%v, kappa=%v\n", tau, kappa)
		if solopts.ShowProgress {
			// show something
            fmt.Printf("%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e\n",
				iter, pcost, dcost, gap, pres, dres)
		}

		if (pres <= solopts.FeasTol && dres <= solopts.FeasTol &&
			(gap <= solopts.AbsTol || (!math.IsNaN(relgap) && relgap <= solopts.RelTol))) ||
			iter == solopts.MaxIter {
			// done
			blas.ScalFloat(x, 1.0/tau.Float())
			blas.ScalFloat(y, 1.0/tau.Float())
			blas.ScalFloat(s, 1.0/tau.Float())
			blas.ScalFloat(z, 1.0/tau.Float())
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
				fmt.Printf("No solution. Max iterations exceeded\n")
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
			blas.ScalFloat(y, 1.0/(-hz - by))
			blas.ScalFloat(z, 1.0/(-hz - by))
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
			blas.ScalFloat(x, 1.0/(-cx))
			blas.ScalFloat(s, 1.0/(-cx))
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
		if iter == 0 {
			fmt.Printf("compute scaling: lmbda=\n%v\ns=\n%v\nz=\n%v\n", lmbda, s, z)
			W, err = ComputeScaling(s, z, lmbda, dims, 0)
			//fmt.Printf("*** initial scaling:\ns:\n%v\nz:\n%v\nlmbda:\n%v\n", s, z, lmbda)
			//fmt.Printf("** W **\n")
			//W.Print()
			//fmt.Printf("** end W **\n")

			//     dg = sqrt( kappa / tau )
			//     dgi = sqrt( tau / kappa )
			//     lambda_g = sqrt( tau * kappa )  
			// 
			// lambda_g is stored in the last position of lmbda.

			dg = math.Sqrt(kappa.Float()/tau.Float())
			dgi = math.Sqrt(float64(tau.Float()/kappa.Float()))
			lmbda.SetIndex(-1, math.Sqrt(float64(tau.Float()*kappa.Float())))
			fmt.Printf("lmbda=\n%v\n", lmbda)
		}
		// lmbdasq := lmbda o lmbda 
		Ssqr(lmbdasq, lmbda, dims, 0)
		lmbdasq.SetIndex(-1, lmbda.GetIndex(-1))
		fmt.Printf("lmbdasq=\n%v\n", lmbdasq)

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


		f3, err = kktsolver(W, nil, nil)
		if err != nil {
			fmt.Printf("kktsolver error=%v\n", err)
			return
		}
		//_, err = kktsolver.Factor(W, nil, nil)
		if iter == 0 {
			x1 = c.Copy()
			y1 = b.Copy()
			z1 = matrix.FloatZeros(cdim, 1)
		}
		blas.Copy(c, x1)
		blas.ScalFloat(x1, -1.0)
		blas.Copy(b, y1)
		blas.Copy(h, z1)
		err = f3(x1, y1, z1)
		fmt.Printf("f3 result: x1=\n%v\nf3 result: z1=\n%v\n", x1, z1)
		//err = kktsolver.Solve(x1, y1, z1)
		blas.ScalFloat(x1, dgi)
		blas.ScalFloat(y1, dgi)
		blas.ScalFloat(z1, dgi)

		if err != nil {
			if iter == 0 && primalstart != nil && dualstart != nil {
				//err = errors.New("Rank(A) < p or Rank([G; A]) < n")
				return
			} else {
				t_ := 1.0/tau.Float()
				blas.ScalFloat(x, t_)
				blas.ScalFloat(y, t_)
				blas.ScalFloat(s, t_)
				blas.ScalFloat(z, t_)
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

        // th = W^{-T} * h
		if iter == 0 {
			th = matrix.FloatZeros(cdim, 1)
		}

		blas.Copy(h, th)
		Scale(th, W, true, true)
		fmt.Printf("th=\n%v\n", th)

		f6_no_ir := func(x, y, z, tau, s, kappa *matrix.FloatMatrix) (err error) {
            // Solve 
            // 
            // [  0   A'  G'    0   ] [ ux        ]   
            // [ -A   0   0     b   ] [ uy        ]  
            // [ -G   0   W'*W  h   ] [ W^{-1}*uz ] 
            // [ -c' -b' -h'    k/t ] [ utau/dg   ]
            // 
            //   [ bx                    ]
            //   [ by                    ]
            // = [ bz - W'*(lmbda o\ bs) ]
            //   [ btau - bkappa/tau     ]
            //
            // us = -lmbda o\ bs - uz
            // ukappa = -bkappa/lmbdag - utau.

            // First solve 
            // 
            // [ 0  A' G'   ] [ ux        ]   [  bx                    ]
            // [ A  0  0    ] [ uy        ] = [ -by                    ]
            // [ G  0 -W'*W ] [ W^{-1}*uz ]   [ -bz + W'*(lmbda o\ bs) ]
			fmt.Printf("== start of f6-no-ir ..\n")
			fmt.Printf("f6-no-ir: x=\n%v\n", x)
			fmt.Printf("f6-no-ir: z=\n%v\n", z)
			fmt.Printf("f6-no-ir: s=\n%v\n", s)
			err = nil
            // y := -y = -by
			blas.ScalFloat(y, -1.0)

            // s := -lmbda o\ s = -lmbda o\ bs
			//fmt.Printf("f6_no_ir: pre-Sinv s=\n%v\nlmbda=\n%v\n", s, lmbda)
			err = Sinv(s, lmbda, dims, 0)
			blas.ScalFloat(s, -1.0)
			//fmt.Printf("f6_no_ir: post-Sinv s=\n%v\nlmbda=\n%v\n", s, lmbda)

            // z := -(z + W'*s) = -bz + W'*(lambda o\ bs)
			blas.Copy(s, ws3)
			//fmt.Printf("f6_no_ir: scaling  ws3=\n%v\n", ws3)
			err = Scale(ws3, W, true, false)
			blas.AxpyFloat(ws3, z, 1.0)
			blas.ScalFloat(z, -1.0)

			//fmt.Printf("f6_no_ir: pre-f3 z=\n%v\n", z)
			fmt.Printf("== calling f3 ...\n")
			err = f3(x, y, z)
			fmt.Printf("== return from f3 ...\n")
			fmt.Printf("x=\n%v\n", x)
			fmt.Printf("z=\n%v\n", z)
			fmt.Printf("== data from f3 ...\n")
			//fmt.Printf("f6_no_ir: post-f3=%v\n", err)
			//err = kktsolver.Solve(x, y, z)

            // Combine with solution of 
            // 
            // [ 0   A'  G'    ] [ x1         ]          [ c ]
            // [-A   0   0     ] [ y1         ] = -dgi * [ b ]
            // [-G   0   W'*W  ] [ W^{-1}*dzl ]          [ h ]
            // 
            // to satisfy
            // 
            // -c'*x - b'*y - h'*W^{-1}*z + dg*tau = btau - bkappa/tau. '

            // kappa[0] := -kappa[0] / lmbd[-1] = -bkappa / lmbdag
			kap_ := kappa.Float()
			tau_ := tau.Float()
			kap_ = -kap_ /lmbda.GetIndex(-1)

            // tau[0] = tau[0] + kappa[0] / dgi = btau[0] - bkappa / tau
			tau_ = tau_ + kap_/dgi

            //tau[0] = dgi * ( tau[0] + xdot(c,x) + ydot(b,y) + 
            //    misc.sdot(th, z, dims) ) / (1.0 + misc.sdot(z1, z1, dims))
			//tau_ = tau_ + blas.DotFloat(c, x) + blas.DotFloat(b, y) + Sdot(th, z, dims, 0)
			tau_ += blas.DotFloat(c, x)
			tau_ += blas.DotFloat(b, y)
			tau_ += Sdot(th, z, dims, 0)
			tau_ = dgi * tau_ / (1.0 + Sdot(z1, z1, dims, 0))
			tau.SetValue(tau_)
			blas.AxpyFloat(x1, x, tau_)
			blas.AxpyFloat(y1, y, tau_)
			blas.AxpyFloat(z1, z, tau_)

			blas.AxpyFloat(z, s, -1.0)
			kap_ = kap_ - tau_
			kappa.SetValue(kap_)
			fmt.Printf("== end of f6-no-ir ..\n")
			return
		}

        // f6(x, y, z, tau, s, kappa) solves the same system as f6_no_ir, 
        // but applies iterative refinement. Following variables part of f6-closure
		// and ~ 12 is the limit. We wrap them to a structure.

		if iter == 0 {
			if refinement > 0 || solopts.Debug {
				WS.wx = c.Copy()
				WS.wy = b.Copy()
				WS.wz = matrix.FloatZeros(cdim, 1)
				WS.ws = matrix.FloatZeros(cdim, 1)
				WS.wtau = matrix.FloatValue(0.0)
				WS.wkappa = matrix.FloatValue(0.0)
			}
			if refinement > 0 {
				WS.wx2 = c.Copy()
				WS.wy2 = b.Copy()
				WS.wz2 = matrix.FloatZeros(cdim, 1)
				WS.ws2 = matrix.FloatZeros(cdim, 1)
				WS.wtau2 = matrix.FloatValue(0.0)
				WS.wkappa2 = matrix.FloatValue(0.0)
			}
		}

		f6 := func(x, y, z, tau, s, kappa *matrix.FloatMatrix) error {
			fmt.Printf("== start of f6 ..\n")
			var err error =  nil
			if refinement > 0 || solopts.Debug {
				blas.Copy(x, WS.wx)
				blas.Copy(y, WS.wy)
				blas.Copy(z, WS.wz)
				blas.Copy(s, WS.ws)
				WS.wtau.SetValue(tau.Float())
				WS.wkappa.SetValue(kappa.Float())
			}
			//fmt.Printf("f6 pre-no-ir: tau=%v, kappa=%v, wx=\n%v\n", tau, kappa, WS.wx)
			err = f6_no_ir(x, y, z, tau, s, kappa)
			fmt.Printf("== return from f6-no-ir ...\n")
			fmt.Printf("x=\n%v\n", x)
			fmt.Printf("z=\n%v\n", z)
			fmt.Printf("s=\n%v\n", s)
			fmt.Printf("== data from f6-no-ir ...\n")
			//fmt.Printf("f6 post-no-ir: tau=%v, kappa=%v\n", tau, kappa)
			for i := 0; i < refinement; i++ {
				blas.Copy(WS.wx, WS.wx2)
				blas.Copy(WS.wy, WS.wy2)
				blas.Copy(WS.wz, WS.wz2)
				blas.Copy(WS.ws, WS.ws2)
				WS.wtau2.SetValue(WS.wtau.Float())
				WS.wkappa2.SetValue(WS.wkappa.Float())
				err = res(x, y, z, tau, s, kappa, WS.wx2, WS.wy2, WS.wz2, WS.wtau2, WS.ws2, WS.wkappa2, W, dg, lmbda)
				//fmt.Printf("f6 refinement 1: tau=%v, kappa=%v, wx2=\n%v\n", i, WS.wtau2, WS.wkappa2, WS.wx2)
				err = f6_no_ir(WS.wx2, WS.wy2, WS.wz2, WS.wtau2, WS.ws2, WS.wkappa2)
				fmt.Printf("== return from f6-no-ir [*]...\n")
				fmt.Printf("wx2=\n%v\n", WS.wx2)
				fmt.Printf("wz2=\n%v\n", WS.wz2)
				fmt.Printf("ws2=\n%v\n", WS.ws2)
				fmt.Printf("== data from f6-no-ir [*]...\n")
				blas.AxpyFloat(WS.wx2, x, 1.0)
				blas.AxpyFloat(WS.wy2, y, 1.0)
				blas.AxpyFloat(WS.wz2, z, 1.0)
				blas.AxpyFloat(WS.ws2, s, 1.0)
				tau.SetValue(tau.Float() + WS.wtau2.Float())
				kappa.SetValue(kappa.Float() + WS.wkappa2.Float())
				//fmt.Printf("%d f6 refinement 2: tau=%v, kappa=%v\n", i, tau, kappa)
			}
			if solopts.Debug {
				res(x, y, z, tau, s, kappa, WS.wx, WS.wy, WS.wz, WS.wtau, WS.ws, WS.wkappa, W, dg, lmbda)
				fmt.Printf("KKT residuals\n")
			}
			fmt.Printf("== end of f6 ..\n")
			return err
		}

		var nrm float64 = blas.Nrm2(lmbda).Float()
        mu := math.Pow(nrm, 2.0) / (1.0 + float64(cdim_diag))
        sigma := 0.0
		var step, tt, tk float64

		for i := 0; i < 2; i++ {
			wkappa3 := matrix.FloatValue(0.0)

            // Solve
            // 
            // [ 0         ]   [  0   A'  G'  c ] [ dx        ]
            // [ 0         ]   [ -A   0   0   b ] [ dy        ]
            // [ W'*ds     ] - [ -G   0   0   h ] [ W^{-1}*dz ]
            // [ dg*dkappa ]   [ -c' -b' -h'  0 ] [ dtau/dg   ]
            // 
            //               [ rx   ]
            //               [ ry   ]
            // = - (1-sigma) [ rz   ]
            //               [ rtau ]
            // 
            // lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e
            // lmbdag * (dtau + dkappa) = - kappa * tau + sigma*mu
            // 
            // ds = -lmbdasq if i is 0
            //    = -lmbdasq - dsa o dza + sigma*mu*e if i is 1
            // dkappa = -lambdasq[-1] if i is 0 
            //        = -lambdasq[-1] - dkappaa*dtaua + sigma*mu if i is 1.
			ind := dims.Sum("l", "q")
			ind2 := ind
			blas.Copy(lmbdasq, ds, &la_.IOpt{"n", ind})
			blas.ScalFloat(ds, 0.0, &la_.IOpt{"offset", ind})
			for _, m := range dims.At("s") {
				blas.Copy(lmbdasq, ds, &la_.IOpt{"n", m}, &la_.IOpt{"offsetx", ind2},
					&la_.IOpt{"offsety", ind}, &la_.IOpt{"incy", m+1})
				ind += m*m
				ind2 += m
			}
			// dkappa[0] = lmbdasq[-1]
			dkappa.SetValue(lmbdasq.GetIndex(-1))
			if i == 1 {
				fmt.Printf("** scaling with sigma*mu (%v, %v)\n", sigma, mu)
				blas.AxpyFloat(ws3, ds, 1.0)
				sigmaMu := func(a float64)float64 {
					return a - sigma*mu
				}
				ind = dims.Sum("l", "q")
				ind2 = ind // ?? NEEDED ??
				// ds[:dims['l']] -= sigma*mu
				ds.ApplyToIndexes(ds, matrix.MakeIndexSet(0, dims.At("l")[0], 1), sigmaMu)
				// ds[indq[:-1]] -= sigma*mu  !! WHAT IS THIS !!
				fmt.Printf("** sigmaMu scaling indexes %v\n", indq[:len(indq)-1])
				ds.ApplyToIndexes(ds, indq[:len(indq)-1], sigmaMu)
				for _, m := range dims.At("s") {
					// ds[ind : ind+m*m : m+1] -= sigma*mu
					iset := matrix.MakeIndexSet(ind, ind+m*m, m+1)
					ds.ApplyToIndexes(ds, iset, sigmaMu)
					ind += m*m
				}
				dk_ := dkappa.Float()
				wk_ := wkappa3.Float()
				dkappa.SetValue(dk_ + wk_ - sigma*mu)
			}
            // (dx, dy, dz, dtau) = (1-sigma)*(rx, ry, rz, rt)
			blas.Copy(rx, dx)
			blas.ScalFloat(dx, 1.0-sigma)
			blas.Copy(ry, dy)
			blas.ScalFloat(dy, 1.0-sigma)
			blas.Copy(rz, dz)
			blas.ScalFloat(dz, 1.0-sigma)
            // dtau[0] = (1.0 - sigma) * rt 
			//dtau = matrix.FloatValue(0.0)
			dtau.SetValue((1.0-sigma)*rt)

			fmt.Printf("== calling f6[%d] ==\n", i)
			fmt.Printf("dx=\n%v\ndz=\n%v\nds=\n%v\n", dx, dz, ds)
			fmt.Printf("dtau=%f, dkappa=%f\n", dtau.Float(), dkappa.Float())
			fmt.Printf("== entering f6[%d] ==\n", i)
			err = f6(dx, dy, dz, dtau, ds, dkappa)
			fmt.Printf("== return from f6[%d] ==\n", i)
			fmt.Printf("dx=\n%v\ndz=\n%v\nds=\n%v\n", dx, dz, ds)
			fmt.Printf("== data from f6[%d] ==\n", i)
			//fmt.Printf("post-f6: dtau=%v, dkappa=%v\n", dtau, dkappa)

			// Save ds o dz and dkappa * dtau for Mehrotra correction
			if i == 0 {
				blas.Copy(ds, ws3)
				Sprod(ws3, dz, dims, 0)
				wkappa3.SetValue(dtau.Float() * dkappa.Float())
			}

            // Maximum step to boundary.
            // 
            // If i is 1, also compute eigenvalue decomposition of the 's' 
            // blocks in ds, dz.  The eigenvectors Qs, Qz are stored in 
            // dsk, dzk.  The eigenvalues are stored in sigs, sigz. 
			var ts, tz float64
			fmt.Printf("pre scale2 lmbda=\n%v\nds=\n%v\n", lmbda, ds)
			Scale2(lmbda, ds, dims, 0, false)
			fmt.Printf("post scale2 lmbda=\n%v\nds=\n%v\n", lmbda, ds)
			Scale2(lmbda, dz, dims, 0, false)
			if i == 0 {
				ts = MaxStep(ds, dims, 0, nil)
				tz = MaxStep(dz, dims, 0, nil)
			} else {
				ts = MaxStep(ds, dims, 0, sigs)
				tz = MaxStep(dz, dims, 0, sigz)
			}
			dt_ := dtau.Float()
			dk_ := dkappa.Float()
			tt = -dt_ / lmbda.GetIndex(-1)
			tk = -dk_ / lmbda.GetIndex(-1)
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
				sigma = math.Pow((1.0 - step), float64(EXPON))
			}
		}
		
		// Update x, y
		blas.AxpyFloat(dx, x, step)
		blas.AxpyFloat(dy, y, step)

        // Replace 'l' and 'q' blocks of ds and dz with the updated 
        // variables in the current scaling.
        // Replace 's' blocks of ds and dz with the factors Ls, Lz in a 
        // factorization Ls*Ls', Lz*Lz' of the updated variables in the 
        // current scaling.
		//
        // ds := e + step*ds for 'l' and 'q' blocks.
        // dz := e + step*dz for 'l' and 'q' blocks.
		blas.ScalFloat(ds, step, &la_.IOpt{"n", dims.Sum("l", "q")})
		blas.ScalFloat(dz, step, &la_.IOpt{"n", dims.Sum("l", "q")})

		addOne := func(v float64)float64 { return v+1.0 }
		ds.ApplyToIndexes(nil, matrix.MakeIndexSet(0, dims.At("l")[0], 1), addOne)
		dz.ApplyToIndexes(nil, matrix.MakeIndexSet(0, dims.At("l")[0], 1), addOne)
		ds.ApplyToIndexes(nil, indq, addOne)
		dz.ApplyToIndexes(nil, indq, addOne)

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
		addOne = func(v float64)float64 {return v+1.0}
        sigs.Apply(nil, addOne)
        sigz.Apply(nil, addOne)
		sdimsum := dims.Sum("q")
		qdimsum := dims.Sum("l", "q")
		blas.TbsvFloat(lmbda, sigs, &la_.IOpt{"n", sdimsum}, &la_.IOpt{"k", 0}, &la_.IOpt{"lda", 1},
			&la_.IOpt{"offseta", qdimsum})
		blas.TbsvFloat(lmbda, sigz, &la_.IOpt{"n", sdimsum}, &la_.IOpt{"k", 0}, &la_.IOpt{"lda", 1},
			&la_.IOpt{"offseta", qdimsum})
		
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
		
		fmt.Printf("pre update_scaling lmbda=\n%v\nds=\n%v\ndz=\n%v\n", lmbda, ds, dz)
		err = UpdateScaling(W, lmbda, ds, dz)
		fmt.Printf("post update_scaling lmbda=\n%v\nds=\n%v\ndz=\n%v\n", lmbda, ds, dz)
        // For kappa, tau block: 
        //
        //     dg := sqrt( (kappa + step*dkappa) / (tau + step*dtau) ) 
        //         = dg * sqrt( (1 - step*tk) / (1 - step*tt) )
        //
        //     lmbda[-1] := sqrt((tau + step*dtau) * (kappa + step*dkappa))
        //                = lmbda[-1] * sqrt(( 1 - step*tt) * (1 - step*tk))
		dg *= math.Sqrt(1.0-step*tk) / math.Sqrt(1.0-step*tt)
		dgi = 1.0/dg
		a := lmbda.GetIndex(-1) + math.Sqrt(1.0-step*tk) / math.Sqrt(1.0-step*tt)
		lmbda.SetIndex(-1, a)
		
        // Unscale s, z, tau, kappa (unscaled variables are used only to 
        // compute feasibility residuals).
		ind := dims.Sum("l", "q")
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
		
		kappa.SetValue(lmbda.GetIndex(-1)/dgi)
		tau.SetValue(lmbda.GetIndex(-1)*dgi)
		g := blas.Nrm2Float(lmbda, &la_.IOpt{"n", lmbda.Rows()-1})/tau.Float()
		gap = g*g
		
	}
	return 
}


// Local Variables:
// tab-width: 4
// End:
