
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package lapack

import (
	"github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/matrix"
	"errors"
)

/*
 Singular value decomposition of a real or complex matrix.

 Gesvd(A, S, jobu=PJobNo, jobvt=PJobNo, U=nil, Vt=nil, m=A.Rows,
 n=A.Cols, ldA=max(1,A.Rows), ldU=-1, ldVt=-1,
 offsetA=0, offsetS=0, offsetU=0, offsetVt=0)

 PURPOSE

 Computes singular values and, optionally, singular vectors of a 
 real or complex m by n matrix A.

 The argument jobu controls how many left singular vectors are
 computed: 

  PJobNo : no left singular vectors are computed.
  PJobAll: all left singular vectors are computed and returned as
           columns of U.
  PJobS  : the first min(m,n) left singular vectors are computed and
           returned as columns of U.
  PJobO  : the first min(m,n) left singular vectors are computed and
           returned as columns of A.
 
 The argument jobvt controls how many right singular vectors are
 computed:

  PJobNo : no right singular vectors are computed.
  PJobAll: all right singular vectors are computed and returned as
           rows of Vt.
  PJobS  : the first min(m,n) right singular vectors are computed and
           returned as rows of Vt.
  PJobO  : the first min(m,n) right singular vectors are computed and
           returned as rows of A.

 Note that the (conjugate) transposes of the right singular 
 vectors are returned in Vt or A.
 On exit (in all cases), the contents of A are destroyed.
 
 ARGUMENTS
  A         float or complex matrix
  S         float matrix of length at least min(m,n).  On exit, 
            contains the computed singular values in descending order.
  jobu      PJobNo, PJobAll, PJobS or PJobO
  jobvt     PJobNo, PJobAll, PJobS or PJobO
  U         float or complex matrix.  Must have the same type as A.
            Not referenced if jobu is PJobNo or PJobO.  If jobu is PJobAll,
            a matrix with at least m columns.   If jobu is PJobS, a
            matrix with at least min(m,n) columns.
            On exit (with jobu PJobAll or PJobS), the columns of U
            contain the computed left singular vectors.
  Vt        float or complex matrix.  Must have the same type as A.
            Not referenced if jobvt is PJobNo or PJobO.  If jobvt is 
            PJobAll or PJobS, a matrix with at least n columns.
            On exit (with jobvt PJobAll or PJobS), the rows of Vt
            contain the computed right singular vectors, or, in
            the complex case, their complex conjugates.
  m         integer.  If negative, the default value is used.
  n         integer.  If negative, the default value is used.
  ldA       nonnegative integer.  ldA >= max(1,m).
            If zero, the default value is used.
  ldU       nonnegative integer.
            ldU >= 1        if jobu is PJobNo or PJobO
            ldU >= max(1,m) if jobu is PJobAll or PJobS.
            The default value is max(1,U.Rows) if jobu is PJobAll 
            or PJobS, and 1 otherwise.
            If zero, the default value is used.
  ldVt      nonnegative integer.
            ldVt >= 1 if jobvt is PJobNo or PJobO.
            ldVt >= max(1,n) if jobvt is PJobAll.  
            ldVt >= max(1,min(m,n)) if ldVt is PJobS.
            The default value is max(1,Vt.Rows) if jobvt is PJobAll
            or PJobS, and 1 otherwise.
            If zero, the default value is used.
  offsetA   nonnegative integer
  offsetS   nonnegative integer
  offsetU   nonnegative integer
  offsetVt  nonnegative integer

 */
func Gesvd(A, S, U, Vt matrix.Matrix, opts ...linalg.Opt) error {
	return errors.New("not implemented yet")
}


// Local Variables:
// tab-width: 4
// End:
