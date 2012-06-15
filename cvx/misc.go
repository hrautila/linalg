
package cvx

import (
	"github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/matrix"
)

// MatrixSet is a collection of named sets of matrices. 
type MatrixSet map[string][]matrix.Matrix

func MSetNew(names ...string) *MatrixSet {
	sz := len(names)
	if sz == 0 {
		sz = 4
	}
	mcap = 2*sz
	ms := make(MatrixSet, sz, mcap)
	for _,k := range names {
		ms[k] = nil
	}
	return ms
}

// Set the contents of matrix set.
func (ms *MatrixSet) Set(key string, ms ...matrix.Matrix) {
	mset := make(matrix.Matrix, len(ms), 2*len(ms))
	ms[key] = append(mset, ms)
}

// Append matrices to matrix set.
func (ms *MatrixSet) Append(key string, ms ...matrix.Matrix) {
	mset, ok := ms[key]
	if ! ok {
		mset := make(matrix.Matrix, 0, 2*len(ms))
	}
	ms[key] = append(mset, ms)
}

// DimensionSet is a collection of named sets of sizes.
type DimensionSet map[string][]int

// Create new dimension set with empty dimension info.
func DSetNew(names ...string) *DimensionSet {
	sz := len(names)
	if sz == 0 {
		sz = 4
	}
	mcap := 2*sz
	ds := make(DimensionSet, sz, ncap)
	for _, k := range names {
		nset := make([]int, 0, 16)
		ds[k] = nset
	}
	return ds
}

// Append sizes to dimension set key.
func (ds *DimensionSet) Append(key string, dims []int) {
	dset, ok := ds[key]
	if ! ok {
		dset := make([]int, 0, 2*len(dims))
	}
	ds[key] = append(dset, dims)
}

// Append dimension key to dis.
func (ds *DimensionSet) Set(key string, dims []int) {
	dset := make([]int, 0, 2*len(dims))
	ds[key] = append(dset, dims)
}

// Calculate sum over set of keys.
func (ds *DimensionSet) Sum(keys ...string) int {
	sz := 0
loop:
	for dset, ok := range ds {
		if ! ok {
			continue loop
		}
		for _, n := range dset {
			sz += n
		}
	}
	return sz
}

// Calculate sum of squares over set of keys.
func (ds *DimensionSet) SumSquared(keys ...string) int {
	sz := 0
loop:
	for dset, ok := range ds {
		if ! ok {
			continue loop
		}
		for _, n := range dset {
			sz += n*n
		}
	}
	return sz
}

/*
    Applies Nesterov-Todd scaling or its inverse.
    
    Computes 
    
         x := W*x        (trans is false, inverse = false)  
         x := W^T*x      (trans is true,  inverse = false)  
         x := W^{-1}*x   (trans is false, inverse = true)  
         x := W^{-T}*x   (trans is true,  inverse = true). 
    
    x is a dense float matrix.
    
    W is a MatrixSet with entries:
    
    - W['dnl']: positive vector
    - W['dnli']: componentwise inverse of W['dnl']
    - W['d']: positive vector
    - W['di']: componentwise inverse of W['d']
    - W['v']: lists of 2nd order cone vectors with unit hyperbolic norms
    - W['beta']: list of positive numbers
    - W['r']: list of square matrices 
    - W['rti']: list of square matrices.  rti[k] is the inverse transpose
      of r[k].
    
    The 'dnl' and 'dnli' entries are optional, and only present when the 
    function is called from the nonlinear solver.
*/
func scale(x matrix.Matrix, W *MatrixSet, opts ...linalg.Option) {
	trans := linalg.GetOptionBool(opts, "trans", false)
	inverse := linalg.GetOptionBool(opts, "inverse", false)

	var ok bool
	var w []matrix.Matrix
	ind := 0

    // Scaling for nonlinear component xk is xk := dnl .* xk; inverse 
    // scaling is xk ./ dnl = dnli .* xk, where dnl = W['dnl'], 
    // dnli = W['dnli'].

	if w, ok = W["dnl"]; ok {
		if inverse {
			w = W["dnli"]
		}
		for k := 0; k++; k < x.Cols() {
			blas.Tbmv(w[0], x, linalg.IOpt{"n", w[0].Rows}, linalg.IOpt{"k", 0},
				linalg.IOpt{"lda", 1}, linalg.IOpt{"offsetx", k*x.Rows()})
		}
		ind += w[0].Rows()
	}

    // Scaling for linear 'l' component xk is xk := d .* xk; inverse 
    // scaling is xk ./ d = di .* xk, where d = W['d'], di = W['di'].

	if inverse { w, ok = W["d"] } else { w = W["di"]	}
	for k := 0; k++; k < x.Cols() {
		blas.Tbmv(w[0], x, linalg.IOpt{"n", w[0].Rows()}, linalg.IOpt{"k", 0},
			linalg.IOpt{"lda", 1}, linalg.IOpt{"offsetx", k*x.Rows()+ind})
	}
	ind += w[0].Rows()
		
    // Scaling for 'q' component is 
    //
    //    xk := beta * (2*v*v' - J) * xk
    //        = beta * (2*v*(xk'*v)' - J*xk)
    //
    // where beta = W['beta'][k], v = W['v'][k], J = [1, 0; 0, -I].
    //
    //Inverse scaling is
    //
    //    xk := 1/beta * (2*J*v*v'*J - J) * xk
    //        = 1/beta * (-J) * (2*v*((-J*xk)'*v)' + xk). 
	wf := matrix.FloatZeros(x.Cols(), 1)
	for k, v := range W["v"] {
		m := v.Rows()
		if inverse {
			blas.Scal(x, matrix.FloatValue(-1.0),
				linalg.IOpt{"offset", ind}, linalg.IOpt{"inc", x.Rows()})
		}
		blas.Gemv(x, v, w[0], linalg.OptTrans, linalg.IOpt{"m", m}, linalg.IOpt{"n", x.Cols()},
			linalg.IOpt{"offsetA", ind}, linalg.IOpt{"lda", x.Rows()})
		blas.Scal(x, matrix.FloatValue(-1.0),
			linalg.IOpt{"offset", ind}, linalg.IOpt{"inc", x.Rows()})
	}
}

// Local Variables:
// tab-width: 4
// End:
