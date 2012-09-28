
package lapack

import (
	"github.com/hrautila/matrix"
	"testing"
	"fmt"
)

// simple matrix implementaion for testing


func TestDGetrf(t *testing.T) {
	ipiv := make([]int32, 3)
	A := matrix.FloatNew(3, 2, []float64{1, 2, 3, 4, 5, 6})
	fmt.Printf("pre A:\n%s\n", A)
	err := Getrf(A, ipiv)
	fmt.Printf("err=%s, ipiv=%s\n", err, ipiv)
	fmt.Printf("post A:\n%s\n", A)
}

func TestFoo(t *testing.T) {
	fmt.Printf("OK\n")
}


// Local Variables:
// tab-width: 4
// End:
