
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/cvx package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.


package cvx

import (
	la_ "github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/linalg/lapack"
	"github.com/hrautila/go.opt/matrix"
	"fmt"
	//"math"
)

func setDiagonal(M *matrix.FloatMatrix, srow, scol, erow, ecol int, val float64) {
	for i := srow; i < erow; i++ {
		if i < ecol {
			M.SetAt(i, i, val)
		}
	}
}

// Solution of KKT equations by a dense LDL factorization of the 
// 3 x 3 system.
//    
// Returns a function that (1) computes the LDL factorization of
//    
// [ H           A'   GG'*W^{-1} ] 
// [ A           0    0          ],
// [ W^{-T}*GG   0   -I          ] 
//    
// given H, Df, W, where GG = [Df; G], and (2) returns a function for 
// solving 
//    
// [ H     A'   GG'   ]   [ ux ]   [ bx ]
// [ A     0    0     ] * [ uy ] = [ by ].
// [ GG    0   -W'*W  ]   [ uz ]   [ bz ]
//    
// H is n x n,  A is p x n, Df is mnl x n, G is N x n where
// N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ).
//
func kktLdl(G *matrix.FloatMatrix, dims *DimensionSet, A *matrix.FloatMatrix, mnl int) (kktFactor, error) {

	p, n := A.Size()
	ldK := n + p + mnl + dims.At("l")[0] + dims.Sum("q") + dims.SumPacked("s")
	K := matrix.FloatZeros(ldK, ldK)
	ipiv := make([]int32, ldK)
	u := matrix.FloatZeros(ldK, 1)
	g := matrix.FloatZeros(mnl+G.Rows(), 1)

	factor := func(W *FloatMatrixSet, H, Df *matrix.FloatMatrix) (KKTFunc, error) {
		var err error = nil
		// Zero K for each call.
		blas.ScalFloat(K, 0.0)
		if H != nil {
			K.SetSubMatrix(0, 0, H)
		}
		K.SetSubMatrix(n, 0, A)
		//fmt.Printf("G=\n%v\n", G)
		for k := 0; k < n; k++ {
			// g is (mnl + G.Rows(), 1) matrix, Df is (mnl, n), G is (N, n)
			if mnl > 0 {
				// set values g[0:mnl] = Df[,k]
				g.SetIndexes(matrix.MakeIndexSet(0, mnl, 1), Df.GetColumnArray(k, nil))
			}
			// set values g[mnl:] = G[,k]
			g.SetIndexes(matrix.MakeIndexSet(mnl, mnl+g.Rows(), 1), G.GetColumnArray(k, nil))
			scale(g, W, true, true)
			if err != nil {
				fmt.Printf("scale error: %s\n", err)
			}
			pack(g, K, dims, &la_.IOpt{"mnl", mnl}, &la_.IOpt{"offsety", k*ldK+n+p})
		}
		setDiagonal(K, n+p, n+n, ldK, ldK, -1.0)
		//fmt.Printf("K=\n%v\n", K)
		err = lapack.Sytrf(K, ipiv)
		//fmt.Printf("sytrf: K=\n%v\n", K)
		if err != nil { return nil, err }

		solve := func(x, y, z *matrix.FloatMatrix) (err error) {
            // Solve
            //
            //     [ H          A'   GG'*W^{-1} ]   [ ux   ]   [ bx        ]
            //     [ A          0    0          ] * [ uy   [ = [ by        ]
            //     [ W^{-T}*GG  0   -I          ]   [ W*uz ]   [ W^{-T}*bz ]
            //
            // and return ux, uy, W*uz.
            //
            // On entry, x, y, z contain bx, by, bz.  On exit, they contain
            // the solution ux, uy, W*uz.
			//fmt.Printf("** start solve **\n")
			//fmt.Printf("x=\n%v\n", x.ConvertToString())
			//fmt.Printf("z=\n%v\n", z.ConvertToString())
			err = nil
			blas.Copy(x, u)
			blas.Copy(y, u, &la_.IOpt{"offsety", n})
			//fmt.Printf("solving: u=\n%v\n", u.ConvertToString())
			//W.Print()
			err = scale(z, W, true, true)
			//fmt.Printf("solving: post-scale z=\n%v\n", z.ConvertToString())
			if err != nil { return }
			err = pack(z, u, dims, &la_.IOpt{"mnl", mnl}, &la_.IOpt{"offsety", n+p})
			//fmt.Printf("solve: post-Pack {mnl=%d, n=%d, p=%d} u=\n%v\n",
			//	mnl, n, p, u.ConvertToString())
			if err != nil { return }

			err = lapack.Sytrs(K, u, ipiv)
			if err != nil {	return }

			blas.Copy(u, x, &la_.IOpt{"n", n})
			blas.Copy(u, y, &la_.IOpt{"n", p}, &la_.IOpt{"offsetx", n})
			err = unpack(u, z, dims, &la_.IOpt{"mnl", mnl}, &la_.IOpt{"offsetx", n+p})
			//fmt.Printf("** end solve **\n")
			//fmt.Printf("x=\n%v\n", x.ConvertToString())
			//fmt.Printf("z=\n%v\n", z.ConvertToString())
			return 
		}
		return solve, err
	}
	return factor, nil
}


type kktLdlSolver struct {
	p, n, ldK, mnl int
	K, u, g *matrix.FloatMatrix
	ipiv []int32
	G, A *matrix.FloatMatrix
	dims *DimensionSet
	W *FloatMatrixSet
	//H, Df *matrix.FloatMatrix
}

// not really needed. 
func createLdlSolver(G *matrix.FloatMatrix, dims *DimensionSet, A *matrix.FloatMatrix, mnl int) *kktLdlSolver {
	kkt := new(kktLdlSolver)
	
	kkt.p, kkt.n = A.Size()
	kkt.ldK = kkt.n + kkt.p + mnl + dims.Sum("l", "q") + dims.SumPacked("s")
	kkt.K = matrix.FloatZeros(kkt.ldK, kkt.ldK)
	kkt.ipiv = make([]int32, kkt.ldK)
	kkt.u = matrix.FloatZeros(kkt.ldK, 1)
	kkt.g = matrix.FloatZeros(kkt.mnl+G.Rows(), 1)
	kkt.G = G
	kkt.A = A
	kkt.dims = dims
	kkt.mnl = mnl
	return kkt
}



// Local Variables:
// tab-width: 4
// End:
