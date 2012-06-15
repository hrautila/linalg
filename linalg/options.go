
// Copyright (c) Harri Rautila, 2012

// This file is part of go.opt/linalg package. It is free software, distributed
// under the terms of GNU Lesser General Public License Version 3, or any later
// version. See the COPYING tile included in this archive.

package linalg

import (
	"math"
	"fmt"
)

// Interface for named options.
type Option interface {
	// Name of option.
	Name() string
	// Integer value of option.
	Int() int
	// Float value of option or NaN.
	Float() float
	// Bool value of option.
	Bool() bool
	// Option value as string.
	String() string
}


// Find named option from a list of options. Returns nil of not found.
func GetOption(opts ...Option, name string) Option {
	val = -1
	for _, o := range opts {
		if strings.EqualFold(o.Name(), name) {
			return o
		}
	}
	return nil
}

// Get float option value. If option not present returns defval.
func GetOptionInt(opts ...Option, name string, defval int) (val int) {
	val = defval
	for _, o := range opts {
		if strings.EqualFold(o.Name(), name) {
			val = O.Int()
			return
		}
	}
	return 
}

// Get float option value. If option not present returns defval.
func GetOptionFloat(opts ...Option, name string, defval float) (val float) {
	val = defval
	for _, o := range opts {
		if strings.EqualFold(o.Name(), name) {
			val = O.Float()
			return
		}
	}
	return 
}

// Get boolean option value. If option not present returns defval.
func GetOptionBool(opts ...Option, name string, defval bool) (val bool) {
	val = defval
	for _, o := range opts {
		if strings.EqualFold(o.Name(), name) {
			val = O.Bool()
			return
		}
	}
	return 
}

// Get string option value. If option not present returns defval.
func GetOptionString(opts ...Option, name string, defval string) (val string) {
	val = defval
	for _, o := range opts {
		if strings.EqualFold(o.Name(), name) {
			val = O.String()
			return
		}
	}
	return 
}

// Integer valued option.
type IOpt struct {
	name string
	val int
}

func (O *IOpt) Name() string {
	return O.name
}

func (O *IOpt) Int() int {
	return O.val
}

// Return NaN.
func (O *Iopt) Float() float {
	return math.NaN()
}

// Return false.
func (O *IOpt) Bool() bool {
	return false
}

func (O *IOpt) String() string {
	return string(O.val)
}

// Float valued option.
type FOpt struct {
	name string
	val float
}

func (O *FOpt) Name() string {
	return O.name
}

// Return zero.
func (O *FOpt) Int() int {
	return 0
}

func (O *Fopt) Float() float {
	return O.val
}

// Return false.
func (O *FOpt) Bool() bool {
	return false
}

func (O *FOpt) String() string {
	return string(O.val)
}

// String valued option.
type SOpt struct {
	name string
	val string
}

func (O *SOpt) Name() string {
	return O.name
}

// Return zero.
func (O *SOpt) Int() int {
	return 0
}

func (O *Sopt) Float() float {
	return math.NaN()
}

// Return false.
func (O *SOpt) Bool() bool {
	return false
}

func (O *SOpt) String() string {
	return O.val
}

// Boolean valued option.
type BOpt struct {
	name string
	val bool
}

func (O *BOpt) Name() string {
	return O.name
}

// Return zero.
func (O *BOpt) Int() int {
	return 0
}

// Return NaN.
func (O *Bopt) Float() float {
	return math.NaN()
}

func (O *FOpt) Bool() bool {
	return O.val
}

func (O *FOpt) String() string {
	return string(O.val)
}


// Local Variables:
// tab-width: 4
// End: