
package cvx

import (
	la_ "github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/linalg/lapack"
	"github.com/hrautila/go.opt/matrix"
	"fmt"
	"math"
)

func setDiagonal(M *matrix.FloatMatrix, srow, scol, erow, ecol int, val float64) {
	for i := srow; i < erow; i++ {
		if i < ecol {
			M.Set(i, i, val)
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
	//fmt.Printf("KktLdl: ldK = %d, p=%d, n=%d\n", ldK, p, n)
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
		for k := 0; k < n; k++ {
			// g is (mnl + G.Rows(), 1) matrix, Df is (mnl, n), G is (N, n)
			if mnl > 0 {
				// set values g[0:mnl] = Df[,k]
				g.SetIndexes(matrix.MakeIndexSet(0, mnl, 1), Df.GetColumn(k, nil))
			}
			// set values g[mnl:] = G[,k]
			g.SetIndexes(matrix.MakeIndexSet(mnl, mnl+g.Rows(), 1), G.GetColumn(k, nil))
			Scale(g, W, true, true)
			if err != nil {
				fmt.Printf("scale error: %s\n", err)
			}
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
			fmt.Printf("** start solve **\n")
			fmt.Printf("solving: x=\n%v\n", x)
			fmt.Printf("solving: z=\n%v\n", z)
			err = nil
			blas.Copy(x, u)
			blas.Copy(y, u, &la_.IOpt{"offsety", n})
			if matrixNaN(u) {
				fmt.Printf("warning!! solver: u has NaN value before Scale!!\n")
			}
			//W.Print()
			err = Scale(z, W, true, true)
			//fmt.Printf("solving: post-scale z=\n%v\n", z)
			if err != nil { return }
			if matrixNaN(z) {
				fmt.Printf("warning!! solver: z has NaN value after Scale!!\n")
				W.Print()
			}
			err = Pack(z, u, dims, &la_.IOpt{"mnl", mnl}, &la_.IOpt{"offsety", n+p})
			if matrixNaN(u) {
				fmt.Printf("warning!! solver: u has NaN value after Pack!!\n")
			}
			if err != nil { return }

			err = lapack.Sytrs(K, u, ipiv)
			if err != nil { return }

			blas.Copy(u, x, &la_.IOpt{"n", n})
			blas.Copy(u, y, &la_.IOpt{"n", p}, &la_.IOpt{"offsetx", n})
			err = UnPack(u, z, dims, &la_.IOpt{"mnl", mnl}, &la_.IOpt{"offsetx", n+p})
			if matrixNaN(x) {
				fmt.Printf("warning!! solver: x has NaN value!!\n")
			}
			fmt.Printf("** end solve **\n")
			return 
		}
		return solve, err
	}
	return factor, nil
}

func matrixNaN(x *matrix.FloatMatrix) bool {
	for i := 0; i < x.NumElements(); i++ {
		if math.IsNaN(x.GetIndex(i)) {
			return true
		}
	}
	return false
}

type KKT interface {
	Factor(W *FloatMatrixSet, H, Df *matrix.FloatMatrix) (KKTFunc, error)
	Solve(x, y, z *matrix.FloatMatrix) error
}

type KKTLdlSolver struct {
	p, n, ldK, mnl int
	K, u, g *matrix.FloatMatrix
	ipiv []int32
	G, A *matrix.FloatMatrix
	dims *DimensionSet
	W *FloatMatrixSet
	//H, Df *matrix.FloatMatrix
}

func CreateLdlSolver(G *matrix.FloatMatrix, dims *DimensionSet, A *matrix.FloatMatrix, mnl int) KKT {
	kkt := new(KKTLdlSolver)
	
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

func (kkt *KKTLdlSolver) GetK() *matrix.FloatMatrix { return kkt.K }
func (kkt *KKTLdlSolver) Getg() *matrix.FloatMatrix { return kkt.g }
func (kkt *KKTLdlSolver) Getu() *matrix.FloatMatrix { return kkt.u }
func (kkt *KKTLdlSolver) Getipiv() []int32 { return kkt.ipiv }

func (kkt *KKTLdlSolver) Factor(W *FloatMatrixSet, H, Df *matrix.FloatMatrix) (KKTFunc, error) {
	var err error = nil
	kkt.W = W

	//fmt.Printf("** kktldl.factor ** \n")
	// Zero K for each call.
	blas.ScalFloat(kkt.K, 0.0)
	if H != nil {
		kkt.K.SetSubMatrix(0, 0, H)
	}
	kkt.K.SetSubMatrix(kkt.n, 0, kkt.A)
	//fmt.Printf("preloop-K:\n%v\n", K)
	for k := 0; k < kkt.n; k++ {
		// g is (mnl + G.Rows(), 1) matrix, Df is (mnl, n), G is (N, n)
		if kkt.mnl > 0 {
			// set values g[0:mnl] = Df[,k]
			kkt.g.SetIndexes(matrix.MakeIndexSet(0, kkt.mnl, 1), Df.GetColumn(k, nil))
		}
		// set values g[mnl:] = G[,k]
		//fmt.Printf("KktLdl.factor: k = %d\n", k)
		kkt.g.SetIndexes(matrix.MakeIndexSet(kkt.mnl, kkt.mnl+kkt.g.Rows(), 1),
			kkt.G.GetColumn(k, nil))
		Scale(kkt.g, W, true, true)
		//fmt.Printf("k=%d, g:\n%v\n", k, kkt.g)
		if err != nil {
			fmt.Printf("scale error: %s\n", err)
		}
		Pack(kkt.g, kkt.K, kkt.dims, &la_.IOpt{"mnl", kkt.mnl},
			&la_.IOpt{"offsety", k*kkt.ldK+kkt.n+kkt.p})
	}
	setDiagonal(kkt.K, kkt.n+kkt.p, kkt.n+kkt.n, kkt.ldK, kkt.ldK, -1.0)
	//fmt.Printf("presytrf-K:\n%v\n", K)
	err = lapack.Sytrf(kkt.K, kkt.ipiv)
	//fmt.Printf("factor: postsytrf-K:\n%v\n", kkt.K)
	return nil, err
}


func (kkt *KKTLdlSolver) Solve(x, y, z *matrix.FloatMatrix) (err error) {
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
	blas.Copy(x, kkt.u)
	blas.Copy(y, kkt.u, &la_.IOpt{"offsety", kkt.n})
	//fmt.Printf("-- solve: pre-scale z:\n%v\n", z)
	err = Scale(z, kkt.W, true, true)
	//fmt.Printf("-- solve: post-scale z:\n%v\n", z)
	if err != nil { return }
	
	err = Pack(z, kkt.u, kkt.dims,
		&la_.IOpt{"mnl", kkt.mnl}, &la_.IOpt{"offsety", kkt.n+kkt.p})
	if err != nil { return }
	//fmt.Printf("-- solve: post-pack u:\n%v\n", kkt.u)
	
	err = lapack.Sytrs(kkt.K, kkt.u, kkt.ipiv)
	if err != nil { return }
	//fmt.Printf("--solve: post-sytrs-K:\n%v\n", kkt.K)
	//fmt.Printf("--solve: post-sytrs-u:\n%v\n", kkt.u)

	blas.Copy(kkt.u, x, &la_.IOpt{"n", kkt.n})
	blas.Copy(kkt.u, y, &la_.IOpt{"n", kkt.p}, &la_.IOpt{"offsetx", kkt.n})
	err = UnPack(kkt.u, z, kkt.dims,
		&la_.IOpt{"mnl", kkt.mnl}, &la_.IOpt{"offsetx", kkt.n+kkt.p})
	//fmt.Printf("-- solve: post-unpack z:\n%v\n", z)
	//fmt.Printf("-- postunpack error:%s\n", err)
	return 
}

// Local Variables:
// tab-width: 4
// End:
