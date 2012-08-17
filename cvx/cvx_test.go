
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/cvx package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.


package cvx

import (
	"github.com/hrautila/go.opt/matrix"
	"fmt"
	"testing"
)

func calcDims(dims *DimensionSet) (int, int, int) {
	cdim := dims.Sum("l", "q") + dims.SumSquared("s")
	cdim_pckd := dims.Sum("l", "q") + dims.SumPacked("s")
	cdim_diag := dims.Sum("l", "q", "s")
	return cdim, cdim_pckd, cdim_diag
}

func makeDSet() *DimensionSet {
	dims := DSetNew("l", "q", "s")
	dims.Set("l", []int{2})
	dims.Set("q", []int{4, 4})
	dims.Set("s", []int{3})
	return dims
}

func makeMatrixSet(dims *DimensionSet) *FloatMatrixSet {
	W := FloatSetNew("d", "di", "v", "beta", "r", "rti")
	dd := dims.At("l")[0]
	W.Set("d", matrix.FloatOnes(dd, 1))
	W.Set("di", matrix.FloatOnes(dd, 1))
	dq := len(dims.At("q"))
	W.Set("beta", matrix.FloatOnes(dq, 1))

	for _, n := range dims.At("q")  {
		vm := matrix.FloatZeros(n, 1)
		vm.SetIndex(0, 1.0)
		W.Append("v", vm)
	}
	for _, n := range dims.At("s") {
		W.Append("r", matrix.FloatIdentity(n))
		W.Append("rti", matrix.FloatIdentity(n))
	}
	return W
}

func TestDSet(t *testing.T) {
	dims := makeDSet()
	cdim, cdim_packd, cdim_diag := calcDims(dims)
	fmt.Printf("cdim = %d\ncdim_packd = %d\ncdim_diag = %d\n", cdim, cdim_packd, cdim_diag)
}

func TestMatrixSet(t *testing.T) {
	dims := makeDSet()
	W := makeMatrixSet(dims)
	for k, m := range W.At("d") {
		fmt.Printf("d[%d]:\n%v\n", k, m)
	}
	for k, m := range W.At("di") {
		fmt.Printf("di[%d]:\n%v\n", k, m)
	}
	for k, m := range W.At("beta") {
		fmt.Printf("beta[%d]:\n%v\n", k, m)
	}
	for k, m := range W.At("v") {
		fmt.Printf("v[%d]:\n%v\n", k, m)
	}
	for k, m := range W.At("r") {
		fmt.Printf("r[%d]:\n%v\n", k, m)
	}
	for k, m := range W.At("rti") {
		fmt.Printf("rti[%d]:\n%v\n", k, m)
	}

		
}

func TestCompile(t *testing.T) {
	fmt.Printf("CVX compiles OK\n")
}

// Local Variables:
// tab-width: 4
// End:
