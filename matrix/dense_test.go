
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
}

func TestFParse2(t *testing.T) {
	fmt.Printf("Test matrix string parsing.\n")
	s2 := "[-7.44e-01  1.11e-01  1.29e+00  2.62e+00 -1.82e+00]" +
		"[ 4.59e-01  7.06e-01  3.16e-01 -1.06e-01  7.80e-01]" +
		"[-2.95e-02 -2.22e-01 -2.07e-01 -9.11e-01 -3.92e-01]" +
		"[-7.75e-01  1.03e-01 -1.22e+00 -5.74e-01 -3.32e-01]" +
		"[-1.80e+00  1.24e+00 -2.61e+00 -9.31e-01 -6.38e-01]"

	A, err := FloatParsePy(s2)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("Py-A :\n%v\n", A)
	// this produces error (column count mismatch)
	s := "[1.0  2.0  3.0 4.0]\n[1.1  2.1 3.1]"
	A, err = FloatParsePy(s)
	if err != nil {
		fmt.Printf("error: %v\n", err)
	}
}

func TestCRandom(t *testing.T) {
	B := FloatUniform(3, 2)
	A := FloatUniformSymmetric(3)
	fmt.Printf("B:\n%v\n", B)
	fmt.Printf("A symmetric:\n%v\n", A)
}

func TestCCopy(t *testing.T) {
	fmt.Printf("Test creating and setting elements.\n")
	A := FloatNew(2, 3, []float64{1,4,2,5,3,6})
	fmt.Printf("A:\n%v\n", A)
	C := A.Copy()
	fmt.Printf("C:\n%v\n", C)
	C.SetAt(0, 1, 10.0*C.GetAt(0, 1))
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
	A.Scale(9.0)
	fmt.Printf("A *= 9.0\n%v\n", A)
	A.Add(-1.0)
	fmt.Printf("A -= 1.0\n%v\n", A)
}

func TestMath2(t *testing.T) {
	m := FloatOnes(8, 1)
	iset := make([]int, 0)
	iset = append(iset, []int{0,1,2}...)
	m.Add(1.0, iset...)
	iset = make([]int, 0)
	iset = append(iset, []int{5,6,7}...)
	m.Add(5.0, iset...)
	fmt.Printf("%v\n", m)
}
func TestFuncs(t *testing.T) {
	fmt.Printf("Test matrix element wise math.\n")
	A := FloatZeros(2, 3)
	AddTwo := func (n float64) float64 {
		return n+2.0
	}
	C := Apply(A, AddTwo)
	fmt.Printf("C = AddTwo(A):\n%v\n", C)
}


func TestIndexing(t *testing.T) {
	A := FloatVector([]float64{0, 1, 2, 3, 4, 5})
	fmt.Printf(" 0: %v\n", A.GetIndex(0))
	fmt.Printf("-1: %v\n", A.GetIndex(-1))
	// this should fail: index out of bounds
	//fmt.Printf(" 6: %v\n", A.GetIndex(6))
	fmt.Printf(" every 2nd: %v\n", A.GetIndexes(Indexes(0, A.NumElements(), 2)))
}

func TestScalars(t *testing.T) {
	f := FScalar(2.0)
	fmt.Printf(" f = %v\n", f)
	fmt.Printf("-f = %v\n", -f)
	z := FScalar(f*f)
	fmt.Printf(" z = %v\n", z)
	
}

func TestArrayCreate(t *testing.T) {
	m := FloatVector([]float64{0,1,2,3,4,5,6,7,8,9})
	b := FloatVector(m.FloatArray()[2:5])
	fmt.Printf("len(m) = %d, len(b) = %d, b=\n%v\n", m.NumElements(), b.NumElements(), b)
}

func TestParseSpe(t *testing.T) {
	s := "{2 3 [3.666666666666667, 3.142857142857143, 4.857142857142857, 4.000000000000000, 5.000000000000000, 6.000000000000000]}"
	m, err := FloatParseSpe(s)
	if err != nil {
		fmt.Printf("parse error: %v\n", err)
	} else {
		fmt.Printf("rows: %d, cols: %d, data:\n%v\n", m.Rows(), m.Cols(), m)
	}

	s2 := "{0 1 []}"
	m, err = FloatParseSpe(s2)
	if err != nil {
		fmt.Printf("parse error: %v\n", err)
	} else {
		fmt.Printf("rows: %d, cols: %d, data:\n%v\n", m.Rows(), m.Cols(), m)
	}
}

func TestStacked(t *testing.T) {
	a := FloatZeros(3,3)
	b := FloatOnes(3,3)
	m0,_ := FloatMatrixStacked(StackDown, a, b)
	m1,_ := FloatMatrixStacked(StackRight, a, b)
	fmt.Printf("stack down=\n%v\n", m0.ToString("%.2f"))
	fmt.Printf("stack right=\n%v\n", m1.ToString("%.2f"))
}

func TestFromTable(t *testing.T) {
	data := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9}}

	a := FloatMatrixFromTable(data, RowOrder)
	b := FloatMatrixFromTable(data, ColumnOrder)
	fmt.Printf("a=\n%v\n", a.ToString("%.2f"))
	fmt.Printf("b=\n%v\n", b.ToString("%.2f"))
	fmt.Printf("b == a:   %v\n", b.Equal(a))
	fmt.Printf("b == a.T: %v\n", b.Equal(a.Transpose()))
}

// Local Variables:
// tab-width: 4
// End:
