
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/cvx package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.


package cvx

import (
	"github.com/hrautila/go.opt/matrix"
	"errors"
	"fmt"
	//"math"
)


//    Solves a pair of primal and dual LPs
//
//        minimize    c'*x
//        subject to  G*x + s = h
//                    A*x = b
//                    s >= 0
//
//        maximize    -h'*z - b'*y
//        subject to  G'*z + A'*y + c = 0
//                    z >= 0.
//
func Lp(c, G, h, A, b *matrix.FloatMatrix, solopts *SolverOptions, primalstart, dualstart *FloatMatrixSet) (sol *Solution, err error) {

	if c == nil {
		err = errors.New("'c' must a column matrix")
		return
	}
	n := c.Rows()
	if n < 1 {
		err = errors.New("Number of variables must be at least 1")
		return
	}
	if G == nil || G.Cols() != n {
		err = errors.New(fmt.Sprintf("'G' must be matrix with %d columns", n))
		return
	}
	m := G.Rows()
	if h == nil || ! h.SizeMatch(m, 1) {
		err = errors.New(fmt.Sprintf("'h' must be matrix of size (%d,1)", m))
		return
	}
	if A == nil {
		A = matrix.FloatZeros(0, n)
	}
	if A.Cols() != n {
		err = errors.New(fmt.Sprintf("'A' must be matrix with %d columns", n))
		return
	}
	p := A.Rows()
	if b == nil {
		b = matrix.FloatZeros(0, 1)
	}
	if ! b.SizeMatch(p, 1) {
		err = errors.New(fmt.Sprintf("'b' must be matrix of size (%d,1)", p))
		return
	}
	dims := DSetNew("l", "q", "s")
	dims.Set("l", []int{m})

	return ConeLp(c, G, h, A, b, dims, solopts, primalstart, dualstart)
}


//    Solves a pair of primal and dual SOCPs
//
//        minimize    c'*x             
//        subject to  Gl*x + sl = hl      
//                    Gq[k]*x + sq[k] = hq[k],  k = 0, ..., N-1
//                    A*x = b                      
//                    sl >= 0,  
//                    sq[k] >= 0, k = 0, ..., N-1
//
//        maximize    -hl'*z - sum_k hq[k]'*zq[k] - b'*y
//        subject to  Gl'*zl + sum_k Gq[k]'*zq[k] + A'*y + c = 0
//                    zl >= 0,  zq[k] >= 0, k = 0, ..., N-1.
//
//    The inequalities sl >= 0 and zl >= 0 are elementwise vector 
//    inequalities.  The inequalities sq[k] >= 0, zq[k] >= 0 are second 
//    order cone inequalities, i.e., equivalent to 
//    
//        sq[k][0] >= || sq[k][1:] ||_2,  zq[k][0] >= || zq[k][1:] ||_2.
//
func Socp(c, Gl, hl, A, b *matrix.FloatMatrix, Ghq *FloatMatrixSet, solopts *SolverOptions, primalstart, dualstart *FloatMatrixSet) (sol *Solution, err error) {
	if c == nil {
		err = errors.New("'c' must a column matrix")
		return
	}
	n := c.Rows()
	if n < 1 {
		err = errors.New("Number of variables must be at least 1")
		return
	}
	if Gl == nil {
		Gl = matrix.FloatZeros(0, n)
	}
	if Gl.Cols() != n {
		err = errors.New(fmt.Sprintf("'G' must be matrix with %d columns", n))
		return
	}
	ml := Gl.Rows()
	if hl == nil {
		hl = matrix.FloatZeros(0, 1)
	}
	if ! hl.SizeMatch(ml, 1) {
		err = errors.New(fmt.Sprintf("'hl' must be matrix of size (%d,1)", ml))
		return
	}
	Gqset := Ghq.At("Gq")
	mq := make([]int, 0)
	for i, Gq := range Gqset {
		if Gq.Cols() != n {
			err = errors.New(fmt.Sprintf("'Gq' must be list of matrices with %d columns", n))
			return
		}
		if Gq.Rows() == 0 {
			err = errors.New(fmt.Sprintf("the number of rows of 'Gq[%d]' is zero", i))
			return
		}
		mq = append(mq, Gq.Rows())
	}
	hqset := Ghq.At("hq")
	if len(Gqset) != len(hqset) {
		err = errors.New(fmt.Sprintf("'hq' must be a list of %d matrices", len(Gqset)))
		return
	}
	for i, hq := range hqset {
		if ! hq.SizeMatch(Gqset[i].Size()) {
			s := fmt.Sprintf("hq[%d] has size (%d,%d). Expected size is (%d,1)",
				i, hq.Rows(), hq.Cols(), Gqset[i].Rows())
			err = errors.New(s)
			return
		}
	}
	if A == nil {
		A = matrix.FloatZeros(0, n)
	}
	if A.Cols() != n {
		err = errors.New(fmt.Sprintf("'A' must be matrix with %d columns", n))
		return
	}
	p := A.Rows()
	if b == nil {
		b = matrix.FloatZeros(0, 1)
	}
	if ! b.SizeMatch(p, 1) {
		err = errors.New(fmt.Sprintf("'b' must be matrix of size (%d,1)", p))
		return
	}
	dims := DSetNew("l", "q", "s")
	dims.Set("l", []int{ml})
	dims.Set("q", mq)
	N := dims.Sum("l", "q")

	h := matrix.FloatZeros(N, 1)
	G := matrix.FloatZeros(N, n)
	h.SetIndexes(matrix.MakeIndexSet(0, ml, 1), hl.FloatArray())
	G.SetSubMatrix(0, 0, Gl)
	//ind := ml
	sol, err = ConeLp(c, G, h, A, b, dims, solopts, nil, nil)
	return
}


// Local Variables:
// tab-width: 4
// End:

