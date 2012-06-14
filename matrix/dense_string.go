
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/matrix package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package matrix

import (
	"strconv"
	"strings"
	"errors"
	"fmt"
)

// Convert matrix to row-major string representation. 
func (A *FloatMatrix) String() string {
	s := ""
	step := A.LeadingIndex()
	for i := 0; i < A.Rows(); i++ {
		if i > 0 {
			s += "\n"
		}
		s += "["
		for j := 0; j < A.Cols(); j++ {
			if j > 0 {
				s += " "
			}
			s += fmt.Sprintf("%f", A.elements[j*step+i])
		}
		s += "]"
	}
	return s
}

// Parse a matlab-style row-major matrix representation eg [a b c; d e f]
// and return a DenseFLoatMatrix.
func FloatParse(s string) (A *FloatMatrix, err error) {
	var arrays [][]float64
	start := strings.Index(s, "[")
	end := strings.LastIndex(s, "]")
	if start == -1 || end == -1 {
		err = errors.New("Unrecognized matrix string")
		return
	}
	rowStrings := strings.Split(s[start+1:end], ";")
	//nrows := len(rowStrings)
	ncols := 0
	for _, row := range rowStrings {
		rowElems := strings.Split(strings.Trim(row, " "), " ")
		if ncols == 0 {
			ncols = len(rowElems)
		} else if ncols != len(rowElems) {
			err = ErrorDimensionMismatch
			return
		}
		row := []float64{}
		for _, valString := range rowElems {
			var val float64
			val, err = strconv.ParseFloat(valString, 64)
			if err != nil {
				return
			}
			row = append(row, val)
		}
		arrays = append(arrays, row)
	}
	A = FloatMatrixStacked(arrays, true)
	return
}

// Local Variables:
// tab-width: 4
// End:
