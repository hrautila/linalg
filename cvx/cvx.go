
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/cvx package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.


// Convex programming package, a port of CVXOPT python package
package cvx

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/cvx/sets"
)

// kktFunc solves
type KKTFunc func(x, y, z *matrix.FloatMatrix) error

// kktFactor produces solver function
type kktFactor func(*sets.FloatMatrixSet, *matrix.FloatMatrix, *matrix.FloatMatrix)(KKTFunc, error)

// kktSolver creates problem spesific factor
type kktSolver func(*matrix.FloatMatrix, *sets.DimensionSet, *matrix.FloatMatrix, int) (kktFactor, error)


type solverMap map[string]kktSolver

var solvers solverMap = solverMap{
	"ldl": kktLdl,
	"ldl2": kktLdl,
	"qr": kktLdl,
	"chol": kktLdl,
	"chol2": kktLdl}


type StatusCode int
const (
	Optimal = StatusCode(1 + iota)
	PrimalInfeasible
	DualInfeasible
	Unknown
)


// If the exit status is 'Optimal', then the primal and dual
// infeasibilities are guaranteed to be less than 
// SolversOptions.FeasTol (default 1e-7).  The gap is less than
// SolversOptions.AbsTol (default 1e-7) or the relative gap is 
// less than SolversOptions.RelTol (defaults 1e-6).     
//
// Termination with status 'Unknown' indicates that the algorithm 
// failed to find a solution that satisfies the specified tolerances.
// In some cases, the returned solution may be fairly accurate.  If
// the primal and dual infeasibilities, the gap, and the relative gap
// are small, then x, y, snl, sl, znl, zl are close to optimal.
//
type Solution struct {
	// Solution status
	Status StatusCode
	// Solution result set. 
	Result *sets.FloatMatrixSet
	// The primal objective c'*x
	PrimalObjective float64
	// The dual objective value
	DualObjective float64
	// Solution duality gap.
	Gap float64
	// Solution relative gap
	RelativeGap float64
	// Solution primal infeasibility
	PrimalInfeasibility float64
	// Solution dual infeasibility: the residual of the dual contraints
	DualInfeasibility float64
	// The smallest primal slack: min( min_k sl_k, sup{t | sl >= te}
	PrimalSlack float64
	// The smallest dual slack: min( min_k sl_k, sup{t | sl >= te}
	DualSlack float64
	PrimalResidualCert float64
	DualResidualCert float64
	// Number of iterations run
	Iterations int
}

// Solver options.
type SolverOptions struct {
	// Absolute tolerance
	AbsTol float64
	// Relative tolerance
	RelTol float64
	// Feasibility tolerance
	FeasTol float64
	// Maximum number of iterations
	MaxIter int
	// Show progress flag
	ShowProgress bool
	// Debug flag
	Debug bool
	// Refinement count
	Refinement int
	// KKT solver function name; 'ldl', 'ldl2', 'qr', 'chol', 'chol2'
	KKTSolverName string
}


const (
	MAXITERS = 100
	ABSTOL = 1e-7
	RELTOL = 1e-6
	FEASTOL = 1e-7
)


// Local Variables:
// tab-width: 4
// End:
