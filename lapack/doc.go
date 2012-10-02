// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg/lapack package.
// It is free software, distributed under the terms of GNU Lesser General Public 
// License Version 3, or any later version. See the COPYING tile included in this archive.

// Interface to the double-precision real and complex LAPACK library.
//
// This package is implementation of CVXOPT Python lapack interface in GO.
//
// Double-precision real and complex LAPACK routines for solving sets of
// linear equations, linear least-squares and least-norm problems,
// symmetric and Hermitian eigenvalue problems, singular value
// decomposition, and Schur factorization.
// 
// For more details, see the LAPACK Users' Guide at
// www.netlib.org/lapack/lug/lapack_lug.html.
//
// Double and complex matrices and vectors are stored in matrices
// using the conventional BLAS storage schemes, with the  matrix
// buffers interpreted as one-dimensional arrays.  For each matrix
// argument X, an additional integer argument offsetX specifies the start
// of the array, i.e., the pointer to X[offsetX:] is passed to the
// LAPACK function.  The other arguments (dimensions and options) have the
// same meaning as in the LAPACK definition.  Default values of the
// dimension arguments are derived from the matrix sizes.
//
// If a routine from the LAPACK library returns with a non zero 'info'
// value function returns with non-nil error with 'info' value included in
// error string.

package lapack
