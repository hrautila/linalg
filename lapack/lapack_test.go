package lapack

import (
    "github.com/hrautila/matrix"
    "testing"
)

// simple matrix implementation for testing

func TestDGetrf(t *testing.T) {
    ipiv := make([]int32, 3)
    A := matrix.FloatNew(3, 2, []float64{1, 2, 3, 4, 5, 6})
    t.Logf("pre A:\n%s\n", A)
    err := Getrf(A, ipiv)
    t.Logf("err=%v, ipiv=%v\n", err, ipiv)
    t.Logf("post A:\n%s\n", A)
}

// Local Variables:
// tab-width: 4
// End:
