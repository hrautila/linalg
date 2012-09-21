
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/cvx package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.


// Convex programming package, a port of CVXOPT python package
package cvx

import (
	"github.com/hrautila/go.opt/matrix"
	la "github.com/hrautila/go.opt/linalg"
)

// Public interface to provide custom G matrix-vector products
//
// The call Gf(u, v, alpha, beta, trans) should evaluate the matrix-vector products
//
//   v := alpha * G * u + beta * v  if trans is linalg.OptNoTrans
//   v := alpha * G' * u + beta * v  if trans is linalg.OptTrans
//
type MatrixG interface {
	Gf(u, v *matrix.FloatMatrix, alpha, beta float64, trans la.Option) error
}
//
// MatrixVarG provides interface for extended customization with non-matrix type
// primal and dual variables.
//
type MatrixVarG interface {
	Gf(u, v MatrixVariable, alpha, beta float64, trans la.Option) error
}


// Public interface to provide custom A matrix-vector products
//
// The call Af(u, v, alpha, beta, trans) should evaluate the matrix-vector products
//
//   v := alpha * A * u + beta * v  if trans is linalg.OptNoTrans
//   v := alpha * A' * u + beta * v  if trans is linalg.OptTrans
//
type MatrixA interface {
	Af(u, v *matrix.FloatMatrix, alpha, beta float64, trans la.Option) error
}
//
// MatrixVarA provides interface for extended customization with non-matrix type
// primal and dual variables.
//
type MatrixVarA interface {
	Af(u, v MatrixVariable, alpha, beta float64, trans la.Option) error
}

// Public interface to provide custom P matrix-vector products
//
// The call Pf(u, v, alpha, beta) should evaluate the matrix-vectors product.
//
//   v := alpha * P * u + beta * v.
//
type MatrixP interface {
	Pf(u, v *matrix.FloatMatrix, alpha, beta float64) error
}

type MatrixVarP interface {
	Pf(u, v MatrixVariable, alpha, beta float64) error
}



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


type MatrixVarDf interface {
	Df(u, v MatrixVariable, alpha, beta float64, trans la.Option) error
}

type MatrixVarH interface {
	Hf(u, v MatrixVariable, alpha, beta float64) error
}

// Empty interface for 
type Variable interface {
	// Convert to string
	String() string
}

// MatrixVariable interface for any type used to represent primal variables
// and the dual variables as something else than one-column float matrices.
//
// If u is an object of type implementing MatrixVariable interface, then
type MatrixVariable interface {
	// Provide internal matrix value
	AsMatrix() *matrix.FloatMatrix
	// Create a new copy 
	Copy() MatrixVariable
	// Computes v := alpha*u + v for a scalar alpha and vectors u and v.
	Axpy(v MatrixVariable, alpha float64) 
	// Return the inner product of two vectors u and v in a vector space.
	Dot(v MatrixVariable) float64
	// Computes u := alpha*u for a scalar alpha and vectors u in a vector space.
	Scal(alpha float64) 
	// Implement checkpnt.Verifiable to allow checkpoint checking
	Verify(vals ...interface{}) float64
	// Convert to string for printing
	String() string
}


// Local Variables:
// tab-width: 4
// End:
