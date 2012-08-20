
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/cvx package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.


package cvx

import (
	"github.com/hrautila/go.opt/matrix"
	"errors"
)

// kktFunc solves
type KKTFunc func(x, y, z *matrix.FloatMatrix) error

// kktFactor produces solver function
type kktFactor func(*FloatMatrixSet, *matrix.FloatMatrix, *matrix.FloatMatrix)(KKTFunc, error)

// kktSolver creates problem spesific factor
type kktSolver func(*matrix.FloatMatrix, *DimensionSet, *matrix.FloatMatrix, int) (kktFactor, error)

func kktNullFactor(W *FloatMatrixSet, H, Df *matrix.FloatMatrix) (KKTFunc, error) {
	nullsolver := func(x, y, z *matrix.FloatMatrix) error {
		return errors.New("Null KTT Solver does not solve anything.")
	}
	return nullsolver, nil
}

func kktNullSolver(G *matrix.FloatMatrix, dims *DimensionSet, A *matrix.FloatMatrix) (kktFactor, error) {
	return kktNullFactor, nil
}

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
	X *matrix.FloatMatrix
	S *matrix.FloatMatrix
	Z *matrix.FloatMatrix
	Y *matrix.FloatMatrix
	Result *FloatMatrixSet
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
