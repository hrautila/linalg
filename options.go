
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package linalg

import (
	"math"
	"math/cmplx"
	"strings"
)

// Interface for named options.
type Option interface {
	// Name of option.
	Name() string
	// Integer value of option.
	Int() int
	// Float value of option or NaN.
	Float() float64
	// Float value of option or NaN.
	Complex() complex128
	// Bool value of option.
	Bool() bool
	// Option value as string.
	String() string
}


// Find named option from a list of options. Returns nil of not found.
func GetOption(name string, opts ...Option) Option {
	for _, o := range opts {
		if strings.EqualFold(o.Name(), name) {
			return o
		}
	}
	return nil
}

// Get integer option value. If option not present returns defval.
func GetIntOpt(name string, defval int, opts ...Option) (val int) {
	val = defval
	for _, o := range opts {
		if strings.EqualFold(o.Name(), name) {
			val = o.Int()
			return
		}
	}
	return 
}

// Get float option value. If option not present returns defval.
func GetFloatOpt(name string, defval float64, opts ...Option) (val float64) {
	val = defval
	for _, o := range opts {
		if strings.EqualFold(o.Name(), name) {
			val = o.Float()
			return
		}
	}
	return 
}

// Get boolean option value. If option not present returns defval.
func GetBoolOpt(name string, defval bool, opts ...Option) (val bool) {
	val = defval
	for _, o := range opts {
		if strings.EqualFold(o.Name(), name) {
			switch o.(type) {
			case *BOpt:
				val = o.Bool()
			case *SOpt:
				v := o.String()
				val = v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y'
			}
			return
		}
	}
	return 
}

// Get string option value. If option not present returns defval.
func GetStringOpt(name string, defval string, opts ...Option) (val string) {
	val = defval
	for _, o := range opts {
		if strings.EqualFold(o.Name(), name) {
			val = o.String()
			return
		}
	}
	return 
}

// Get string option value. If option not present returns defval.
func GetComplexOpt(name string, defval complex128, opts ...Option) (val complex128) {
	val = defval
	for _, o := range opts {
		if strings.EqualFold(o.Name(), name) {
			val = o.Complex()
			return
		}
	}
	return 
}

// Integer valued option.
type IOpt struct {
	OptName string
	Val int
}

// Return integer valued option.
func IntOpt(name string, val int) *IOpt {
	return &IOpt{name, val}

}

func (O *IOpt) Name() string {
	return O.OptName
}

func (O *IOpt) Int() int {
	return O.Val
}

// Return NaN.
func (O *IOpt) Float() float64 {
	return math.NaN()
}

// Return NaN.
func (O *IOpt) Complex() complex128 {
	return cmplx.NaN()
}

// Return false.
func (O *IOpt) Bool() bool {
	return false
}

func (O *IOpt) String() string {
	return string(O.Val)
}

// Float valued option.
type FOpt struct {
	OptName string
	Val float64
}

// Return integer valued option.
func FloatOpt(name string, val float64) *FOpt {
	return &FOpt{name, val}

}

func (O *FOpt) Name() string {
	return O.OptName
}

// Return zero.
func (O *FOpt) Int() int {
	return 0
}

func (O *FOpt) Float() float64 {
	return O.Val
}

// Return NaN.
func (O *FOpt) Complex() complex128 {
	return cmplx.NaN()
}
// Return false.
func (O *FOpt) Bool() bool {
	return false
}

func (O *FOpt) String() string {
	return "" //string(O.val)
}

// String valued option.
type SOpt struct {
	OptName string
	Val string
}

// Return string valued option.
func StringOpt(name string, val string) *SOpt {
	return &SOpt{name, val}

}
func (O *SOpt) Name() string {
	return O.OptName
}

// Return zero.
func (O *SOpt) Int() int {
	return 0
}

func (O *SOpt) Float() float64 {
	return math.NaN()
}

// Return NaN.
func (O *SOpt) Complex() complex128 {
	return cmplx.NaN()
}

// Return false.
func (O *SOpt) Bool() bool {
	return false
}

func (O *SOpt) String() string {
	return O.Val
}

// Boolean valued option.
type BOpt struct {
	OptName string
	Val bool
}

// Return bool valued option.
func BoolOpt(name string, val bool) *BOpt {
	return &BOpt{name, val}
}

func (O *BOpt) Name() string {
	return O.OptName
}

// Return zero.
func (O *BOpt) Int() int {
	return 0
}

// Return NaN.
func (O *BOpt) Float() float64 {
	return math.NaN()
}

// Return NaN.
func (O *BOpt) Complex() complex128 {
	return cmplx.NaN()
}

func (O *BOpt) Bool() bool {
	return O.Val
}

func (O *BOpt) String() string {
	return "" //string(O.val)
}

// Return complex valued option.
func ComplexOpt(name string, val complex128) *COpt {
	return &COpt{name, val}
}

// Complex valued option.
type COpt struct {
	OptName string
	Val complex128
}

func (O *COpt) Name() string {
	return O.OptName
}

// Return zero.
func (O *COpt) Int() int {
	return 0
}

// Return NaN.
func (O *COpt) Float() float64 {
	return math.NaN()
}

// Return NaN.
func (O *COpt) Complex() complex128 {
	return O.Val
}

func (O *COpt) Bool() bool {
	return false
}

func (O *COpt) String() string {
	return "" //string(O.val)
}


// Local Variables:
// tab-width: 4
// End: