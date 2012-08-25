
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import (
	"testing"
	"fmt"
)

func asMatrix(A Matrix) {
	fmt.Printf("Test linalg.Matrix methods.\n")
	M := A.MakeCopy()
	fmt.Printf("M size: %d rows, %d cols\n", M.Rows(), M.Cols())
	fmt.Printf("M elems: %d\n", M.NumElements())
	fmt.Printf("M:\n%v\n", M)
}

func TestA(t *testing.T) {
	fmt.Printf("Test complex matrix printing.\n")
	A := ComplexZeros(2,2)
	fmt.Printf("A:\n%v\n", A)
}

func TestCParse(t *testing.T) {
	fmt.Printf("Test matrix string parsing.\n")
	s := `[(1.0+0i) (+2-1i) (3.0+0i); ( 4.2-.5i) (-5 -.1i) (6+0i)]`
	A, err := ComplexParse(s)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("A :\n%v\n", A)
	fmt.Printf("A size: %d rows, %d cols\n", A.Rows(), A.Cols())
	fmt.Printf("A elems: %d\n", A.NumElements())
	D := A.Transpose()
	fmt.Printf("D = A.transpose:\n%v\n", D)
	r, c := D.Size()
	fmt.Printf("D size: %d rows, %d cols\n", r, c)
	asMatrix(A)
}

func TestRand(t *testing.T) {
	fmt.Printf("Test matrix creation.\n")
	A := ComplexUniform(3, 2)
	fmt.Printf("A :\n%v\n", A)
	B := ComplexUniformSymmetric(2)
	fmt.Printf("B :\n%v\n", B)
}
	
/*
func TestCopy(t *testing.T) {
	fmt.Printf("Test creating and setting elements.\n")
	A := MakeComplexMatrix(2, 3, []float64{1,4,2,5,3,6})
	fmt.Printf("A:\n%v\n", A)
	C := A.Copy()
	fmt.Printf("C:\n%v\n", C)
	C.Set(0, 1, C.Get(0, 1)*10)
	B := MakeComplexMatrix(3, 2, []float64{1,2,3,4,5,6})
	fmt.Printf("B:\n%v\n", B)
}

func TestBool(t *testing.T) {
	fmt.Printf("Test matrix boolean operators.\n")
	A := MakeComplexMatrix(3, 2, []float64{1,4,2,5,3,6})
	fmt.Printf("A:\n%v\n", A)
	B := A.Copy()
	//B := MakeComplexMatrix(3, 2, []float64{1,2,3,4,5,6})
	fmt.Printf("B:\n%v\n", B)
	fmt.Printf("A == B: %v\n", A.Equal(B))
	fmt.Printf("A != B: %v\n", A.NotEqual(B))
}

func TestMath(t *testing.T) {
	fmt.Printf("Test matrix basic math.\n")
	A := Zeros(2, 2)
	fmt.Printf("A\n%v\n", A)
	A.Add(1.0)
	fmt.Printf("A += 1.0\n%v\n", A)
	A.Mult(9.0)
	fmt.Printf("A *= 9.0\n%v\n", A)
	A.Sub(1.0)
	fmt.Printf("A -= 1.0\n%v\n", A)
	A.Div(2.0)
	fmt.Printf("A /= 2.0\n%v\n", A)
	A.Remainder(3.0)
	fmt.Printf("A %%= 3.0\n%v\n", A)
	A.Neg()
	fmt.Printf("A = -A:\n%v\n", A)
	C := A.Times(A)
	fmt.Printf("C = A*A:\n%v\n", C)
	D := C.Plus(A)
	fmt.Printf("D = C+A:\n%v\n", D)
	F := D.Minus(A)
	fmt.Printf("F = D-A:\n%v\n", F)
	G := Zeros(3, 2); G.Add(1.0)
	H := G.Copy().Transpose()
	fmt.Printf("G:\n%v\n", G)
	fmt.Printf("H:\n%v\n", H)
	K := G.Times(H)
	fmt.Printf("K = G*H:\n%v\n", K)
}

func TestFuncs(t *testing.T) {
	fmt.Printf("Test matrix element wise math.\n")
	A := Zeros(2, 3)
	AddTwo := func (n float64) float64 {
		return n+2.0
	}
	C := A.Apply(AddTwo)
	fmt.Printf("C = AddTwo(A):\n%v\n", C)
}

func TestFuncs2(t *testing.T) {
	A := Ones(2, 3)
	B := Exp(A)
	fmt.Printf("B = Exp(A):\n%v\n", B)
}
*/

// Local Variables:
// tab-width: 4
// End:
