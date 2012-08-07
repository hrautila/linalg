
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

	
// BLAS/LAPACK matrix parameter constants.
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
	PDiag = ParamValue(133)				// 'D'
	PLeft = ParamValue(141)				// 'L'
	PRight = ParamValue(142)			// 'R'
	// These for LAPACK only
	PJobNo = ParamValue(151)			// 'N'
	PJobValue = ParamValue(152)			// 'V'
	PJobAll = ParamValue(153)			// 'A'
	PJobS = ParamValue(154)				// 'S'
	PJobO = ParamValue(155)				// 'O'
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
	Jobu ParamValue
	Jobvt ParamValue
	Range ParamValue
}

func GetParam(name string, params ...Option) (val int) {
	val = -1
	for _, o := range params {
		if strings.EqualFold(o.Name(), name) {
			val = o.Int()
			return
		}
	}
	return 
}

// Parse options and return parameter structure with option fields
// set to given or sensible defaults.
func GetParameters(params ...Option) (p *Parameters, err error) {
	err = nil
	p = &Parameters{
		PNoTrans,		// Trans
		PNoTrans,		// TransA
		PNoTrans,		// TransB
		PLower,			// Uplo
		PNonUnit,		// Diag
		PLeft,			// Side
		PJobNo,			// Jobz
		PJobNo,			// Jobu
		PJobNo,			// Jobvt
		PRangeAll}		// Range

Loop:
	for _, o := range params {
		if _, ok := o.(*IOpt); ! ok {
			continue Loop
		}
		pval := ParamValue(o.Int())
		switch {
		case strings.EqualFold(o.Name(), "trans"):
			if pval == PNoTrans || pval == PTrans || pval == PConjTrans {
				p.Trans = pval
				p.TransA = p.Trans;	p.TransB = p.Trans
			} else {
				err = errors.New("Illegal value for Transpose parameter")
				break Loop
			}
		case strings.EqualFold(o.Name(), "transa"):
			if pval == PNoTrans || pval == PTrans || pval == PConjTrans {
				p.TransA = pval
			} else {
				err = errors.New("Illegal value for Transpose parameter")
				break Loop
			}
		case strings.EqualFold(o.Name(), "transb"):
			if pval == PNoTrans || pval == PTrans || pval == PConjTrans {
				p.TransB = pval
			} else {
				err = errors.New("Illegal value for Transpose parameter")
				break Loop
			}
		case strings.EqualFold(o.Name(), "uplo"):
			if pval == PUpper || pval == PLower {
				p.Uplo = pval
			} else {
				err = errors.New("Illegal value for UpLo parameter")
				break Loop
			}
		case strings.EqualFold(o.Name(), "diag"):
			if pval == PNonUnit || pval == PUnit || pval == PDiag {
				p.Diag = pval
			} else {
				err = errors.New("Illegal value for Diag parameter")
				break Loop
			}
		case strings.EqualFold(o.Name(), "side"):
			if pval ==  PLeft || pval == PRight {
				p.Side = pval
			} else {
				err = errors.New("Illegal value for Side parameter")
				break Loop
			}
		// Lapack parameters
		case strings.EqualFold(o.Name(), "jobz"):
			if pval == PJobNo || pval == PJobValue {
				p.Jobz = pval
			} else {
				err = errors.New("Illegal value for Jobz parameter")
				break Loop
			}
		case strings.EqualFold(o.Name(), "jobu"):
			if pval == PJobNo || pval == PJobAll || pval == PJobS || pval == PJobO {
				p.Jobu = pval
			} else {
				err = errors.New("Illegal value for Jobu parameter")
				break Loop
			}
		case strings.EqualFold(o.Name(), "jobvt"):
			if pval == PJobNo || pval == PJobAll || pval == PJobS || pval == PJobO {
				p.Jobvt = pval
			} else {
				err = errors.New("Illegal value for Jobu parameter")
				break Loop
			}
		case strings.EqualFold(o.Name(), "range"):
			if pval == PRangeAll || pval == PRangeValue || pval == PRangeInt {
				p.Range = pval
			} else {
				err = errors.New("Illegal value for Range parameter")
				break Loop
			}
		}
	}
	return
}

// Matrix parameter option variables.
var (
	// trans: No Transpose
	OptNoTrans = &IOpt{"trans", int(PNoTrans)}
	OptNoTransA = &IOpt{"transA", int(PNoTrans)}
	OptNoTransB = &IOpt{"transB", int(PNoTrans)}
	// trans: Transpose
	OptTrans = &IOpt{"trans", int(PTrans)}
	OptTransA = &IOpt{"transA", int(PTrans)}
	OptTransB = &IOpt{"transB", int(PTrans)}
	// trans: Conjugate Transpose
	OptConjTrans = &IOpt{"trans", int(PConjTrans)}
	OptConjTransA = &IOpt{"transA", int(PConjTrans)}
	OptConjTransB = &IOpt{"transB", int(PConjTrans)}
	// uplo: Upper Triangular
	OptUpper = &IOpt{"uplo", int(PUpper)}
	// uplo: Lower Triangular
	OptLower = &IOpt{"uplo", int(PLower)}
	// side parameter
	OptLeft = &IOpt{"side", int(PLeft)}
	OptRight = &IOpt{"side", int(PRight)}
	// diag parameter
	OptUnit =  &IOpt{"diag", int(PUnit)}
	OptNonUnit =  &IOpt{"diag", int(PNonUnit)}
	OptDiag =  &IOpt{"diag", int(PDiag)}
	// Lapack jobz 
	OptJobZNo =  &IOpt{"jobz", int(PJobNo)}
	OptJobZValue =  &IOpt{"jobz", int(PJobValue)}
	// Lapack jobu
	OptJobuNo =  &IOpt{"jobu", int(PJobNo)}
	OptJobuAll =  &IOpt{"jobu", int(PJobAll)}
	OptJobuS =  &IOpt{"jobu", int(PJobS)}
	OptJobuO =  &IOpt{"jobu", int(PJobO)}
	// Lapack jobvt
	OptJobvtNo =  &IOpt{"jobvt", int(PJobNo)}
	OptJobvtAll =  &IOpt{"jobvt", int(PJobAll)}
	OptJobvtS =  &IOpt{"jobvt", int(PJobS)}
	OptJobvtO =  &IOpt{"jobvt", int(PJobO)}
	// Lapack range
	OptRangeAll =  &IOpt{"range", int(PRangeAll)}
	OptRangeValue =  &IOpt{"range", int(PRangeValue)}
	OptRangeInt =  &IOpt{"range", int(PRangeInt)}
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
	PJobAll: "A",
	PJobS: "S",
	PJobO: "O",
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
