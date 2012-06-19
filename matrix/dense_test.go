
package matrix

import (
	"testing"
	"fmt"
)

func PrintArray(m Matrix) {
	ar := m.FloatArray()
	fmt.Printf("matrix: %d, %d [%d elements]\n",
		m.Rows(), m.Cols(), m.NumElements())
	fmt.Printf("data: %v\n", ar)
}

func TestFParse(t *testing.T) {
	fmt.Printf("Test matrix string parsing.\n")
	s := `[1.0 2.0 3.0; 4.0 5.0 6.0]`
	A, err := FloatParse(s)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("A :\n%v\n", A)
	D := A.Transpose()
	fmt.Printf("A.transpose:\n%v\n", D)
}

func TestCRandom(t *testing.T) {
	B := FloatRandom(3, 2, true)
	A := FloatRandomSymmetric(3, false)
	fmt.Printf("B:\n%v\n", B)
	fmt.Printf("A symmetric:\n%v\n", A)
}

func TestCCopy(t *testing.T) {
	fmt.Printf("Test creating and setting elements.\n")
	A := FloatNew(2, 3, []float64{1,4,2,5,3,6})
	fmt.Printf("A:\n%v\n", A)
	C := A.Copy()
	fmt.Printf("C:\n%v\n", C)
	C.Set(0, 1, C.Get(0, 1)*10)
	B := FloatNew(3, 2, []float64{1,2,3,4,5,6})
	fmt.Printf("B:\n%v\n", B)
}

func TestFBool(t *testing.T) {
	fmt.Printf("Test matrix boolean operators.\n")
	A := FloatNew(3, 2, []float64{1,4,2,5,3,6})
	fmt.Printf("A:\n%v\n", A)
	B := A.Copy()
	//B := MakeMatrix(3, 2, []float64{1,2,3,4,5,6})
	fmt.Printf("B:\n%v\n", B)
	fmt.Printf("A == B: %v\n", A.Equal(B))
	fmt.Printf("A <  B: %v\n", A.Less(B))
	fmt.Printf("A <= B: %v\n", A.LessOrEqual(B))
	fmt.Printf("A >  B: %v\n", A.Greater(B))
	fmt.Printf("A >= B: %v\n", A.GreaterOrEqual(B))
}

func TestFMath(t *testing.T) {
	fmt.Printf("Test matrix basic math.\n")
	A := FloatZeros(2, 2)
	fmt.Printf("A\n%v\n", A)
	A.Add(1.0)
	fmt.Printf("A += 1.0\n%v\n", A)
	A.Mult(9.0)
	fmt.Printf("A *= 9.0\n%v\n", A)
	A.Sub(1.0)
	fmt.Printf("A -= 1.0\n%v\n", A)
	A.Div(2.0)
	fmt.Printf("A /= 2.0\n%v\n", A)
	A.Mod(3.0)
	fmt.Printf("A %%= 3.0\n%v\n", A)
	A.Neg()
	fmt.Printf("A = -A:\n%v\n", A)
	C := A.Times(A)
	fmt.Printf("C = A*A:\n%v\n", C)
	D := C.Plus(A)
	fmt.Printf("D = C+A:\n%v\n", D)
	F := D.Minus(A)
	fmt.Printf("F = D-A:\n%v\n", F)
	G := FloatZeros(3, 2); G.Add(1.0)
	H := G.Copy().Transpose()
	fmt.Printf("G:\n%v\n", G)
	fmt.Printf("H:\n%v\n", H)
	K := G.Times(H)
	fmt.Printf("K = G*H:\n%v\n", K)
}

func TestFuncs(t *testing.T) {
	fmt.Printf("Test matrix element wise math.\n")
	A := FloatZeros(2, 3)
	AddTwo := func (n float64) float64 {
		return n+2.0
	}
	C := A.Apply(A.Copy(), AddTwo)
	fmt.Printf("C = AddTwo(A):\n%v\n", C)
}

func TestFuncs2(t *testing.T) {
	A := FloatOnes(2, 3)
	B := Exp(A)
	fmt.Printf("B = Exp(A):\n%v\n", B)
}

func TestIndexing(t *testing.T) {
	A := FloatVector([]float64{0, 1, 2, 3, 4, 5})
	fmt.Printf(" 0: %v\n", A.GetIndex(0))
	fmt.Printf("-1: %v\n", A.GetIndex(-1))
	fmt.Printf(" 6: %v\n", A.GetIndex(6))
	fmt.Printf(" every 2nd: %v\n", A.GetIndexes(MakeIndexSet(0, A.NumElements(), 2)))
}

// Local Variables:
// tab-width: 4
// End:
