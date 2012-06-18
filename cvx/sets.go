
package cvx

import (
	//"github.com/hrautila/go.opt/linalg"
	"github.com/hrautila/go.opt/matrix"
)


// FloatMatrixSet is a collection of named sets of float valued matrices. 
type FloatMatrixSet struct {
	sets map[string][]*matrix.FloatMatrix
}

// Create new FloatMatrix collection with with empty named sets. 
func FloatSetNew(names ...string) *FloatMatrixSet {
	sz := len(names)
	if sz == 0 {
		sz = 4
	}
	mcap := 2*sz
	ms := new(FloatMatrixSet)
	ms.sets = make(map[string][]*matrix.FloatMatrix, mcap)
	for _,k := range names {
		ms.sets[k] = nil
	}
	return ms
}

// Get named set
func (M *FloatMatrixSet) At(name string) []*matrix.FloatMatrix {
	mset, _ := M.sets[name]
	return mset
}

// Set the contents of matrix set.
func (M *FloatMatrixSet) Set(key string, ms ...*matrix.FloatMatrix) {
	mset := make([]*matrix.FloatMatrix, len(ms), 2*len(ms))
	for _, v := range ms {
		M.sets[key] = append(mset, v)
	}
}

// Append matrices to matrix set.
func (M *FloatMatrixSet) Append(key string, ms ...*matrix.FloatMatrix) {
	mset, ok := M.sets[key]
	if ! ok {
		mset = make([]*matrix.FloatMatrix, 0, 2*len(ms))
	}
	for _, v := range ms {
		M.sets[key] = append(mset, v)
	}
}


// DimensionSet is a collection of named sets of sizes.
type DimensionSet struct {
	sets map[string][]int
}

// Create new dimension set with empty dimension info.
func DSetNew(names ...string) *DimensionSet {
	sz := len(names)
	if sz == 0 {
		sz = 4
	}
	mcap := 2*sz
	
	ds := new(DimensionSet)
	ds.sets = make(map[string][]int, mcap)
	for _, k := range names {
		nset := make([]int, 0, 16)
		ds.sets[k] = nset
	}
	return ds
}

// Get named set
func (ds *DimensionSet) At(name string) []int {
	dset, _ := ds.sets[name]
	return dset
}

// Append sizes to dimension set key.
func (ds *DimensionSet) Append(key string, dims []int) {
	dset, ok := ds.sets[key]
	if ! ok {
		dset = make([]int, 0, 2*len(dims))
	}
	for _, v := range dims {
		ds.sets[key] = append(dset, v)
	}
}

// Append dimension key to dis.
func (ds *DimensionSet) Set(key string, dims []int) {
	dset := make([]int, 0, 2*len(dims))
	for _, v := range dims {
		ds.sets[key] = append(dset, v)
	}
}

// Calculate sum over set of keys.
func (ds *DimensionSet) Sum(keys ...string) int {
	sz := 0
	for _, key := range keys {
		dset := ds.sets[key]
		for _, n := range dset {
			sz += n
		}
	}
	return sz
}

// Calculate sum of squared dimensions over set of keys.
func (ds *DimensionSet) SumSquared(keys ...string) int {
	sz := 0
	for _, key := range keys {
		dset := ds.sets[key]
		for _, n := range dset {
			sz += n*n
		}
	}
	return sz
}

// Calculate sum of packed dimensions over set of keys.
func (ds *DimensionSet) SumPacked(keys ...string) int {
	sz := 0
	for _, key := range keys {
		dset := ds.sets[key]
		for _, n := range dset {
			sz += n*(n+1)/2
		}
	}
	return sz
}

// Local Variables:
// tab-width: 4
// End:
