
package cvx

import (
	la_ "github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/linalg/lapack"
	"github.com/hrautila/go.opt/matrix"
)

func setDiagonal(M *matrix.FloatMatrix, srow, scol, erow, ecol int, val float64) {
	for i := srow; i < erow; i++ {
		for j := scol; j < ecol; j++ {
			M.Set(i, j, val)
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
func KktLdl(G *matrix.FloatMatrix, dims *DimensionSet, A *matrix.FloatMatrix, mnl int) (KKTFactor, error) {

	p, n := A.Size()
	ldK := n + p + mnl + dims.At("l")[0] + dims.Sum("q") + dims.SumPacked("s")
	K := matrix.FloatZeros(ldK, ldK)
	ipiv := make([]int32, ldK)
	u := matrix.FloatZeros(ldK, 1)
	g := matrix.FloatZeros(mnl+G.Rows(), 1)
	
	factor := func(W *FloatMatrixSet, H, Df *matrix.FloatMatrix) (KKTFunc, error) {

		var err error = nil
		// Zero K for each call.
		blas.Scal(K, matrix.FScalar(0.0))
		if H != nil {
			K.SetSubMatrix(0, 0, H)
		}
		K.SetSubMatrix(n, 0, A)
		for k := 0; k < n; k++ {
			// g is (mnl + G.Rows(), 1) matrix, Df is (mnl, n), G is (N, n)
			if mnl > 0 {
				// set values g[0:mnl] = Df[,k]
				g.SetIndexes(matrix.MakeIndexSet(0, mnl, 1), Df.GetColumn(k, nil))
			}
			// set values g[mnl:] = G[,k]
			g.SetIndexes(matrix.MakeIndexSet(mnl, g.Rows(), 1), G.GetColumn(k, nil))
			Scale(g, W, true, true)
			Pack(g, K, dims, &la_.IOpt{"mnl", mnl}, &la_.IOpt{"offsety", k*ldK+n+p})
		}
		setDiagonal(K, n+p, n+n, ldK, ldK, -1.0)
		err = lapack.Sytrf(K, ipiv)
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
			err = nil
			blas.Copy(x, u)
			blas.Copy(y, u, &la_.IOpt{"offsety", n})
			err = Scale(z, W, true, true)
			if err != nil { return }

			err = Pack(z, u, dims, &la_.IOpt{"mnl", mnl}, &la_.IOpt{"offsety", n+p})
			if err != nil { return }

			err = lapack.Sytrs(K, u, ipiv)
			if err != nil { return }

			blas.Copy(u, x, &la_.IOpt{"n", n})
			blas.Copy(u, y, &la_.IOpt{"n", p}, &la_.IOpt{"offsetx", n})
			err = UnPack(u, z, dims, &la_.IOpt{"mnl", mnl}, &la_.IOpt{"offsetx", n+p})
			return 
		}
		return solve, err
	}
	return factor, nil
}

// Local Variables:
// tab-width: 4
// End:
