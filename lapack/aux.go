// Copyright (c) Harri Rautila, 2013

// This file is part of  package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package lapack

import (
	"github.com/hrautila/linalg"
	"github.com/hrautila/matrix"
	//"errors"
	"fmt"
)

/*
 * LARFT forms the triangular factor T of a real block reflector H
 * of order n, which is defined as a product of k elementary reflectors.
 *
 * Option DIRECT:
 * If DIRECT = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular;
 *
 * If DIRECT = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular.
 *
 * Option STOREV:
 * If STOREV = 'C', the vector which defines the elementary reflector
 * H(i) is stored in the i-th column of the array V, and
 *
 *    H  =  I - V * T * V**T
 *
 * If STOREV = 'R', the vector which defines the elementary reflector
 * H(i) is stored in the i-th row of the array V, and
 *
 *    H  =  I - V**T * T * V

 */
func LarftFloat(V, tau, T *matrix.FloatMatrix, opts ...linalg.Option) {
	var K, N int
	N = V.Rows()
	K = T.Cols()
	direct := linalg.GetStringOpt("direct", "F", opts...)
	storev := linalg.GetStringOpt("storev", "C", opts...)
	if storev[0] == 'C' {
		K = V.Cols()
	} else {
		N = V.Cols()
	}
	ldt := T.LeadingIndex()
	ldv := V.LeadingIndex()
	Vr := V.FloatArray()
	taur := tau.FloatArray()
	Tr := T.FloatArray()
	dlarft(direct, storev, N, K, Vr, ldv, taur, Tr, ldt)
}

func LarfFloat(V, tau, C *matrix.FloatMatrix, opts ...linalg.Option) {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	ind := linalg.GetIndexOpts(opts...)
	if ind.M < 0 {
		ind.M = C.Rows()
	}
	if ind.N < 0 {
		ind.N = C.Cols()
	}
	if ind.LDc == 0 {
		ind.LDc = C.LeadingIndex()
	}
	// column matrix!
	incv := 1
	side := linalg.ParamString(pars.Side)

	Vr := V.FloatArray()
	tr := tau.FloatArray()
	Cr := C.FloatArray()
	dlarf(side, ind.M, ind.N, Vr, incv, tr, Cr, ind.LDc)
}

func LarfbFloat(V, T, C *matrix.FloatMatrix, opts ...linalg.Option) {
	pars, err := linalg.GetParameters(opts...)
	if err != nil {
		return
	}
	direct := linalg.GetStringOpt("direct", "F", opts...)
	storev := linalg.GetStringOpt("storev", "C", opts...)

	ldt := T.LeadingIndex()
	ldv := V.LeadingIndex()
	ind := linalg.GetIndexOpts(opts...)
	if ind.M < 0 {
		ind.M = C.Rows()
	}
	if ind.N < 0 {
		ind.N = C.Cols()
		if storev[0] != 'C' {
			ind.N = V.Cols()
		}
	}
	if ind.K < 0 {
		ind.K = T.Cols()
		if storev[0] == 'C' {
			ind.K = V.Cols()
		}
	}
	if ind.LDc == 0 {
		ind.LDc = C.LeadingIndex()
	}
	// column matrix!
	side := linalg.ParamString(pars.Side)
	trans := linalg.ParamString(pars.Trans)
	Vr := V.FloatArray()
	Tr := T.FloatArray()
	Cr := C.FloatArray()
	dlarfb(side, trans, direct, storev, ind.M, ind.N, ind.K, Vr, ldv, Tr, ldt, Cr, ind.LDc)
}

func LarfgFloat(alpha, X, tau *matrix.FloatMatrix, opts ...linalg.Option) {
}

/*
 */
func OrgqrFloat(A, tau *matrix.FloatMatrix, opts ...linalg.Option) error {
	ind := linalg.GetIndexOpts(opts...)
	if ind.M < 0 {
		ind.M = A.Rows()
	}
	if ind.N < 0 {
		ind.N = A.Cols()
	}
	if ind.K < 0 {
		ind.K = A.Cols()
	}
	if ind.LDa == 0 {
		ind.LDa = A.LeadingIndex()
	}
	Ar := A.FloatArray()
	tr := tau.FloatArray()
	info := dorgqr(ind.M, ind.N, ind.K, Ar, ind.LDa, tr)
	if info != 0 {
		return onError(fmt.Sprintf("Orgqr lapack error: %d", info))
	}
	return nil
}

// Local Variables:
// tab-width: 4
// End:
