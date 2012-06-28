
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import (
	"math"
	"math/rand"
	"math/cmplx"
	"errors"
)

// Find maximum element value for matrix. For complex matrix returns NaN.
func Max(A Matrix) float64 {
	m := math.Inf(-1)
	if ar := A.FloatArray(); ar != nil {
		for _, v := range ar {
			m = math.Max(m, v)
		}
	} else {
		m = math.NaN()
	}
	return m
}

// Find minimum element value for matrix. For complex matrix returns NaN.
func Min(A Matrix) float64 {
	m := math.Inf(+1)
	if ar := A.FloatArray(); ar != nil {
		for _, v := range A.FloatArray() {
			m = math.Min(m, v)
		}
	} else {
		m = math.NaN()
	}
	return m
}

// Return copy of A with each element as Sqrt(A[i,j]).
func Sqrt(A Matrix) Matrix {
	B := A.MakeCopy()
	applyFunc(B, math.Sqrt, cmplx.Sqrt)
	return B
}

// Return copy of A with each element as Exp(A[i,j]).
func Exp(A Matrix) Matrix {
	B := A.MakeCopy()
	applyFunc(B, math.Exp, cmplx.Exp)
	return B
}

// Return copy of A with each element as Log(A[i,j]).
func Log(A Matrix) Matrix {
	B := A.MakeCopy()
	applyFunc(B, math.Log, cmplx.Log)
	return B
}

// Return copy of A with each element as Log10(A[i,j]).
func Log10(A Matrix) Matrix {
	B := A.MakeCopy()
	applyFunc(B, math.Log10, cmplx.Log10)
	return B
}

// Return copy of A with each element as Log1p(A[i,j]). Returns nil for
// complex valued matrix.
func Log1p(A Matrix) Matrix {
	if A.IsComplex() {
		return nil
	}
	B := A.MakeCopy()
	applyFunc(B, math.Log1p, nil)
	return B
}

// Return copy of A with each element as Log2(A[i,j]). Returns nil for
// complex valued matrix.
func Log2(A Matrix) Matrix {
	if A.IsComplex() {
		return nil
	}
	B := A.MakeCopy()
	applyFunc(B, math.Log2, nil)
	return B
}

// Return copy of A with each element as Abs(A[i,j]). Returns a float matrix for
// complex valued matrix.
func Abs(A Matrix) Matrix {
	if ! A.IsComplex() {
		B := A.MakeCopy()
		applyFunc(B, math.Abs, nil)
		return B
	} 
	// Complex matrix here
	B := FloatZeros(A.Size())
	Ba := B.FloatArray()
	Aa := A.ComplexArray()
	for k, v := range Aa {
		Ba[k] = cmplx.Abs(v)
	}
	return B
}

// Return conjugate copy of A with each element as Conj(A[i,j]). Returns nil for
// float valued matrix.
func Conj(A Matrix) Matrix {
	if ! A.IsComplex() {
		return nil
	}
	B := A.MakeCopy()
	applyFunc(B, nil, cmplx.Conj)
	return B
}

// Return copy of A with each element as Pow(A[i,j], y). 
func Pow(A Matrix, y Scalar) Matrix {
	C := A.MakeCopy()
	if C.IsComplex() {
		applyComplex(C, cmplx.Pow, y.Complex())
	} else {
		applyFloat(C, math.Pow, y.Float())
	}
	return C
}

// Return new matrix which is element wise division A / B.
// A and B matrices must be of same type and have equal number of elements.
func Div(A, B Matrix) (Matrix, error) {
	res := A.MakeCopy()
	if A.NumElements() != B.NumElements() {
		return nil, errors.New("Mismatching sizes")
	}
	if ! res.EqualTypes(B) {
		return nil, errors.New("Mismatching types")
	}
	if res.IsComplex() {
		ar := res.ComplexArray()
		br := B.ComplexArray()
		for i, _ := range ar {
			ar[i] /= br[i]
		}
	} else {
		ar := res.FloatArray()
		br := B.FloatArray()
		for i, _ := range ar {
			ar[i] /= br[i]
		}
	}
	return res, nil
}

// Return new matrix which is element wise product of argument matrices.
// A and B matrices must be of same type and have equal number of elements.
func Mul(ml ...Matrix) (Matrix, error) {
	fst := ml[0]
	res := fst.MakeCopy()
	for _, m := range ml[1:] {
		if fst.NumElements() != m.NumElements() {
			return nil, errors.New("Mismatching sizes")
		}
		if ! fst.EqualTypes(m) {
			return nil, errors.New("Mismatching types")
		}
		mul(res, m)
	}
	return res, nil
}


func mul(a, b Matrix) {
	if a.IsComplex() {
		ar := a.ComplexArray()
		br := b.ComplexArray()
		for i, _ := range ar {
			ar[i] *= br[i]
		}
	} else {
		ar := a.FloatArray()
		br := b.FloatArray()
		for i, _ := range ar {
			ar[i] *= br[i]
		}
	}
}

// Return new matrix which is element wise sum of argument matrices.
// A and B matrices must be of same type and have equal number of elements.
func Sum(ml ...Matrix) (Matrix, error) {
	fst := ml[0]
	res := fst.MakeCopy()
	for _, m := range ml[1:] {
		if fst.NumElements() != m.NumElements() {
			return nil, errors.New("Mismatching sizes")
		}
		if ! fst.EqualTypes(m) {
			return nil, errors.New("Mismatching types")
		}
		add(res, m)
	}
	return res, nil
}

func add(a, b Matrix) {
	if a.IsComplex() {
		ar := a.ComplexArray()
		br := b.ComplexArray()
		for i, _ := range ar {
			ar[i] += br[i]
		}
	} else {
		ar := a.FloatArray()
		br := b.FloatArray()
		for i, _ := range ar {
			ar[i] += br[i]
		}
	}
}

func applyFunc(A Matrix, rFunc func(float64)float64, cFunc func(complex128)complex128) {
	if ! A.IsComplex() && rFunc != nil {
		vals := A.FloatArray()
		for i, _ := range vals {
			vals[i] = rFunc(vals[i])
		}
	} else if cFunc != nil {
		vals := A.ComplexArray()
		for i, _ := range vals {
			vals[i] = cFunc(vals[i])
		}
	}
}

func applyComplex(A Matrix, cFunc func(complex128,complex128)complex128, cval complex128) {
	if cFunc != nil {
		vals := A.ComplexArray()
		for i, _ := range vals {
			vals[i] = cFunc(vals[i], cval)
		}
	}
}

func applyFloat(A Matrix, cFunc func(float64,float64)float64, cval float64) {
	if cFunc != nil {
		vals := A.FloatArray()
		for i, _ := range vals {
			vals[i] = cFunc(vals[i], cval)
		}
	}
}

// Helper function for genearing random float64 numbers in range [0.0, 1.0) if nonNegative is true.
// If nonNegative is false returns numbers in range (-1.0, 1.0).
func uniformFloat64(nonNegative bool) float64 {
	val := rand.Float64()
	if nonNegative {
		return val
	}
	return 2.0*(val - 0.5)
}


// Local Variables:
// tab-width: 4
// End:
