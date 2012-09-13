
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

type Solution struct {
	Status StatusCode
	//X *matrix.FloatMatrix
	//Y *matrix.FloatMatrix
	//S *matrix.FloatMatrix
	//Z *matrix.FloatMatrix
	Result *sets.FloatMatrixSet
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

const (
	MAXITERS = 100
	ABSTOL = 1e-7
	RELTOL = 1e-6
	FEASTOL = 1e-7
)


// Local Variables:
// tab-width: 4
// End:
