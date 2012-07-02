
package cvx

import (
	"github.com/hrautila/go.opt/matrix"
	"fmt"
	"testing"
)

func TestMatrixSet(t *testing.T) {
	dims := DSetNew("l", "q", "s")
	dims.Set("l", []int{2})
	dims.Set("q", []int{4, 4})
	dims.Set("s", []int{3})

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
		W.Append("r", matrix.FloatIdentity(n, n))
		W.Append("rti", matrix.FloatIdentity(n, n))
	}

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

func CTest(t *testing.T) {
	fmt.Printf("CVX compiles OK\n")
}

// Local Variables:
// tab-width: 4
// End:
