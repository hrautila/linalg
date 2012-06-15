
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package linalg

import (
	"strings"
	"errors"
	"fmt"
)


type ParamValue uint32
const (
	// BLAS/LAPACK parameters. Chosen values match corresponding
	// parameters in CBLAS implementation.
	RowMajor = ParamValue(101)			// Atlas row-major
	ColumnMajor = ParamValue(102)		// Atlas column major
	PNoTrans = ParamValue(111)			// 'N'
	PTrans = ParamValue(112)			// 'T'
	PConjTrans = ParamValue(113)		// 'C'
	PUpper = ParamValue(121)			// 'U'
	PLower = ParamValue(122)			// 'L'
	PNonUnit = ParamValue(131)			// 'N'
	PUnit = ParamValue(132)				// 'U'
	PLeft = ParamValue(141)				// 'L'
	PRight = ParamValue(142)			// 'R'
	// These for LAPACK only
	PJobNo = ParamValue(151)			// 'N'
	PJobValue = ParamValue(152)			// 'V'
	PRangeAll = ParamValue(161)			// 'A'
	PRangeValue = ParamValue(162)		// 'V'
	PRangeInt = ParamValue(163)			// 'I'
)

// Structure for BLAS/LAPACK function parameters.
type Parameters struct {
	Trans, TransA, TransB ParamValue
	Uplo ParamValue
	Diag ParamValue
	Side ParamValue
	Jobz ParamValue
	Range ParamValue
}

func GetParam(params ...Opt, name string) (val int) {
	val = -1
	for _, o := range params {
		if strings.EqualFold(o.name, name) {
			val = o.Val
			return
		}
	}
	return 
}

// Parse options and return parameter structure with option fields
// set to given or sensible defaults.
func GetParameters(params ...Opt) (p *Parameters, err error) {
	err = nil
	p = &Parameters{
		PNoTrans,		// Trans
		PNoTrans,		// TransA
		PNoTrans,		// TransB
		PLower,			// Uplo
		PUnit,			// Diag
		PLeft,			// Side
		PJobNo,			// Jobz
		PRangeAll}		// Range

Loop:
	for _, o := range params {
		pval := ParamValue(o.Val)
		switch {
		case strings.EqualFold(o.Name, "trans"):
			if pval == PNoTrans || pval == PTrans || pval == PConjTrans {
				p.Trans = pval
				p.TransA = p.Trans;	p.TransB = p.Trans
			} else {
				err = errors.New("Illegal value for Transpose parameter")
				break Loop
			}
		case strings.EqualFold(o.Name, "transa"):
			if pval == PNoTrans || pval == PTrans || pval == PConjTrans {
				p.TransA = pval
			} else {
				err = errors.New("Illegal value for Transpose parameter")
				break Loop
			}
		case strings.EqualFold(o.Name, "transb"):
			if pval == PNoTrans || pval == PTrans || pval == PConjTrans {
				p.TransB = pval
			} else {
				err = errors.New("Illegal value for Transpose parameter")
				break Loop
			}
		case strings.EqualFold(o.Name, "uplo"):
			if pval == PUpper || pval == PLower {
				p.Uplo = pval
			} else {
				err = errors.New("Illegal value for UpLo parameter")
				break Loop
			}
		case strings.EqualFold(o.Name, "diag"):
			if pval == PNonUnit || pval == PUnit {
				p.Diag = pval
			} else {
				err = errors.New("Illegal value for Diag parameter")
				break Loop
			}
		case strings.EqualFold(o.Name, "side"):
			if pval ==  PLeft || pval == PRight {
				p.Side = pval
			} else {
				err = errors.New("Illegal value for Side parameter")
				break Loop
			}
		// Lapack parameters
		case strings.EqualFold(o.Name, "jobz"):
			if pval == PJobNo || pval == PJobValue {
				p.Side = pval
			} else {
				err = errors.New("Illegal value for Jobz parameter")
				break Loop
			}
		case strings.EqualFold(o.Name, "range"):
			if pval == PRangeAll || pval == PRangeValue || pval == PRangeInt {
				p.Side = pval
			} else {
				err = errors.New("Illegal value for Range parameter")
				break Loop
			}
		}
	}
	return
}

// Parameter option variables.
var (
	OptNoTrans = Opt{"trans", int(PNoTrans)}
	OptTrans = Opt{"trans", int(PTrans)}
	OptConjTrans = Opt{"trans", int(PConjTrans)}
	OptNoTransA = Opt{"transA", int(PNoTrans)}
	OptTransA = Opt{"transA", int(PTrans)}
	OptConjTransA = Opt{"transA", int(PConjTrans)}
	OptNoTransB = Opt{"transB", int(PNoTrans)}
	OptTransB = Opt{"transB", int(PTrans)}
	OptConjTransB = Opt{"transB", int(PConjTrans)}
	OptUpper = Opt{"uplo", int(PUpper)}
	OptLower = Opt{"uplo", int(PLower)}
	OptLeft = Opt{"side", int(PLeft)}
	OptRight = Opt{"side", int(PRight)}
	OptUnit =  Opt{"diag", int(PUnit)}
	OptNonUnit =  Opt{"diag", int(PNonUnit)}
	OptJobNo =  Opt{"jobz", int(PJobNo)}
	OptJobValue =  Opt{"jobz", int(PJobValue)}
	OptRangeAll =  Opt{"range", int(PRangeAll)}
	OptRangeValue =  Opt{"range", int(PRangeValue)}
	OptRangeInt =  Opt{"range", int(PRangeInt)}
)

var paramString map[ParamValue]string = map[ParamValue]string{
	PNoTrans: "N",
	PTrans: "T",
	PConjTrans: "C",
	PUpper: "U",
	PLower: "L",
	PLeft: "L",
	PRight: "R",
	PUnit: "U",
	PNonUnit: "N",
	PJobNo: "N",
	PJobValue: "V",
	PRangeAll: "A",
	PRangeValue: "V",
	PRangeInt: "I",
}
	
// Map parameter value to name string that can be used when calling Fortran
// library functions.
func ParamString(p ParamValue) string {
	v, ok := paramString[p]
	if ok {
		return v
	}
	return ""
}

// Print parameter structure.
func PrintParameters(p *Parameters) {
	fmt.Printf("trans : %d [%s]\n", p.Trans, ParamString(p.Trans))
	fmt.Printf("transA: %d [%s]\n", p.TransA, ParamString(p.TransA))
	fmt.Printf("transB: %d [%s]\n", p.TransB, ParamString(p.TransB))
	fmt.Printf("Uplo  : %d [%s]\n", p.Uplo, ParamString(p.Uplo))
	fmt.Printf("Diag  : %d [%s]\n", p.Diag, ParamString(p.Diag))
	fmt.Printf("Side  : %d [%s]\n", p.Side, ParamString(p.Side))
	fmt.Printf("Jobz  : %d [%s]\n", p.Jobz, ParamString(p.Jobz))
	fmt.Printf("Range : %d [%s]\n", p.Range, ParamString(p.Range))
}


// Local Variables:
// tab-width: 4
// End:
