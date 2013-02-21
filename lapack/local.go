// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/linalg/lapack package.
// It is free software, distributed under the terms of GNU Lesser General Public 
// License Version 3, or any later version. See the COPYING tile included in this archive.

package lapack

import "errors"

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func max(a, b int) int {
    if a < b {
        return b
    }
    return a
}

var panicOnError bool = false

func PanicOnError(flag bool) {
	panicOnError = flag
}

func onError(msg string) error {
	if panicOnError {
		panic(msg)
	}
	return errors.New(msg)
}

// Local Variables:
// tab-width: 4
// End:
