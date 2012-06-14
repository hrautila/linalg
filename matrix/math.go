
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import (
	"math"
	"math/rand"
	"math/cmplx"
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

// Return copy of A with each element as Log10(A[i,j]).
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

// Return copy of A with each element as Abs(A[i,j]). Returns nil for
// complex valued matrix.
func Abs(A Matrix) Matrix {
	if A.IsComplex() {
		return nil
	}
	B := A.MakeCopy()
	applyFunc(B, math.Abs, nil)
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

// Return copy of A with each element as Pow(A[i,j], y). For complex matrix
// y is complex(y, 0.0).
func PowF(A Matrix, y float64) Matrix {
	C := A.MakeCopy()
	if C.IsComplex() {
		applyComplex(C, cmplx.Pow, complex(y, 0.0))
	} else {
		applyFloat(C, math.Pow, y)
	}
	return C
}

// Return copy of A with each element as Pow(A[i,j], y). For float matrix
// returns nil.
func PowC(A Matrix, y complex128) Matrix {
	if ! A.IsComplex() {
		return nil
	}
	C := A.MakeCopy()
	applyComplex(C, cmplx.Pow, y)
	return C
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

// Helper function random float64 number in range [0.0, 1.0) if nonNegative is true.
// If nonNegative is false returns numbers in range (-1.0, 1.0).
func uniformFloat64(nonNegative bool) float64 {
	val := rand.Float64()
	if nonNegative {
		return val
	}
	return 2.0*(val - 0.5)
	/*
	n := rand.Int63()
	if n & 1 != 0 {
		val = -val
	}
	return val
	 */
}

// Local Variables:
// tab-width: 4
// End:
