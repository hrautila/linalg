
package checkpnt

import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/linalg/blas"
	"github.com/hrautila/go.opt/cvx/sets"
	"fmt"
	"os"
	"bufio"
	"strings"
	"strconv"
	"errors"
)

type checkpoint struct {
	name string
	filepath string
	major int
	minor int
}

type dataPoint struct {
	mtx *matrix.FloatMatrix
	w *sets.FloatMatrixSet
	fvar *float64
	panicVar bool
	inErrorM bool
	inErrorF bool
	ckp *checkpoint
}

type variableTable map[string]*dataPoint

var variables variableTable
var active bool = false
var spmajor int
var sppath string
var normError float64
var diffError float64
var verbose bool
var minorstack []int
var minorpointer int
var spformat string

func init() {
	variables = make(variableTable, 20)
	spmajor = 0
	sppath = "./"
	normError = 1e-15
	diffError = 1e-12
	verbose = false
	active = false
	minorstack = make([]int, 20)
	minorpointer = 0
	spformat = "%12.5e"
}

// Return current major number.
func Major() int {
	return spmajor
}

// Advance major number by one.
func MajorNext() {
	if active {
		spmajor += 1
	}
}

// Push new minor number on to stack.
func MinorPush(minor int) {
	if ! active {
		return
	}
	if minorpointer == len(minorstack) {
		// stack full
		return
	}
	minorstack[minorpointer] = minor
	minorpointer += 1
}

// Pop minor number on top of the stack.
func MinorPop() int {
	if ! active {
		return -1
	}
	if minorpointer == 0 {
		// stack empty
		return -1
	}
	minorpointer -= 1
	return minorstack[minorpointer]
}

// Get minor number on top of the stack.
func MinorTop() int {
	if ! active {
		return -1
	}
	if minorpointer == 0 {
		// stack empty, 
		return -1
	}
	return minorstack[minorpointer-1]
}

// Test if minor number stack is empty.
func MinorEmpty() bool {
	return minorpointer == 0
}

// Add matrix variable as checkpointable variable.
func AddMatrixVar(name string, mtx *matrix.FloatMatrix) {
	if ! active {
		return
	}
	_, ok := variables[name]
	if ! ok {
		variables[name] = &dataPoint{mtx:mtx}
	}
}

// Set or unset panic flag for variable.
func PanicVar(name string, ispanic bool) {
	if ! active {
		return
	}
	_, ok := variables[name]
	if ok {
		variables[name].panicVar = ispanic
	}
}

// Add float variable as check point variable.
func AddFloatVar(name string, fptr *float64) {
	if ! active {
		return
	}
	if _, ok := variables[name]; ! ok {
		variables[name] = &dataPoint{fvar:fptr}
	}
}

// Add float variable as check point variable.
func AddCpVar(name string, mtx *matrix.FloatMatrix, fptr *float64) {
	if ! active {
		return
	}
	if _, ok := variables[name]; ! ok {
		variables[name] = &dataPoint{mtx:mtx, fvar:fptr}
	}
}

// Add scaling matrix set to checkpoint variables.
func AddScaleVar(w *sets.FloatMatrixSet) {
	if ! active {
		return
	}
	// add all matrices of scale set to variable table
	for _, key := range w.Keys() {
		mset := w.At(key)
		for k, m := range mset {
			name := fmt.Sprintf("%s.%d", key, k)
			if _, ok := variables[name]; ! ok {
				variables[name] = &dataPoint{mtx:m}
			}
		}
	}
}

func UpdateMatrixVar(name string, mtx *matrix.FloatMatrix) {
	if ! active {
		return
	}
	variables[name] = &dataPoint{mtx:mtx}
}

// Print checkpoint variables.
func PrintVariables() {
	for name := range variables {
		dp := variables[name]
		if dp.mtx != nil {
			fmt.Printf("'%s' matrix (%d, %d)\n", name, dp.mtx.Rows(), dp.mtx.Cols())
		} else if dp.w != nil {
			fmt.Printf("'%s' matrix set \n", name)
		}
	}
}

// Report on check point variables. Prints out the last check point variable turned invalid.
func Report() {
	if ! active {
		return
	}
	for name := range variables {
		dp := variables[name]
		if dp.inErrorM {
			fmt.Printf("%8s invalidated at %d.%04d %-12s [%s]\n",
				name, dp.ckp.major, dp.ckp.minor, dp.ckp.name, dp.ckp.filepath)
		}
		if dp.inErrorF {
			fname := name
			if dp.mtx != nil && dp.fvar != nil {
				fname = name + ".t"
			}
			fmt.Printf("%8s invalidated at %d.%04d %-12s [%s]\n",
				fname, dp.ckp.major, dp.ckp.minor, dp.ckp.name, dp.ckp.filepath)
		}
	}
}
	
func Format(format string) {
	spformat = format
}

func Reset(path string) {
	for name := range variables {
		delete(variables, name)
	}
	spmajor = 0
	active = false
	sppath = path
}

func Activate() {
	active = true
}

func Verbose(flag bool) {
	verbose = flag
}

// Check variables at checkpoint.
func Check(name string, minor int) {
	if ! active {
		return
	}
	vars, checkp, err := readCkp(name, minor)
	if err != nil {
		//fmt.Printf("error when reading savepoint '%s': %v\n", name, err)
		return
	}
	// loop through all our savepoint variables
	for varname := range variables {
		dp, ok := (*vars)[varname]		
		if ! ok {
			// varname not found in savepoint reference data
			continue
		}
		// internal data value
		mydp := variables[varname]
		if dp.mtx != nil && dp.fvar == nil {
			// std matrix
			checkMatrix(varname, checkp, dp, mydp)
		} else if dp.fvar != nil && dp.mtx == nil {
			// std float
			checkFloat(varname, checkp, dp, mydp)
		} else if dp.fvar != nil && dp.mtx != nil {
			// cp epigraph
			checkMatrix(varname, checkp, dp, mydp)
			checkFloat(varname+".t", checkp, dp, mydp)
		}
	}
}

func checkFloat(varname string, ckp *checkpoint, refdp, mydp *dataPoint) {
	// std float
	df := *mydp.fvar - *refdp.fvar
	if df > diffError {
		if ! mydp.inErrorF {
			fmt.Printf("%d.%d sp '%s'[file:%s] variable '%s': diff = %9.2e\n",
				ckp.major, ckp.minor, ckp.name, ckp.filepath, varname, df)
			mydp.ckp = ckp
		}
		if verbose && ! mydp.inErrorF || mydp.panicVar {
			fmt.Printf("variable '%s' internal|refrence|difference\n", varname)
			fmt.Printf("%.17f %.17f %.17f\n", *mydp.fvar, *refdp.fvar, df)
		}
		mydp.inErrorF = true
		if mydp.panicVar {
			panic("variable divergence error ...")
		}
	} else {
		if  mydp.inErrorF {
			fmt.Printf("%d.%d sp '%s'[file:%s] variable '%s': returned to valid\n",
				ckp.major, ckp.minor, ckp.name, ckp.filepath, varname)
			mydp.ckp = nil
		}
		mydp.inErrorF = false
	}
}

func checkMatrix(varname string, ckp *checkpoint, refdp, mydp *dataPoint) {
	//fmt.Printf("checking matrix %s ...\n", varname)
	var refval, myval *matrix.FloatMatrix
	refval = refdp.mtx
	myval = mydp.mtx
	dval := matrix.Minus(myval, refval)
	norm := blas.Nrm2Float(dval)
	if norm > normError {
		if ! mydp.inErrorM {
			fmt.Printf("%d.%d sp '%s'[file:%s] variable '%s': normError = %9.2e\n",
				ckp.major, ckp.minor, ckp.name, ckp.filepath, varname, norm)
			mydp.ckp = ckp
		}
		if verbose && ! mydp.inErrorM || mydp.panicVar {
			rm, _ := matrix.FloatMatrixStacked(matrix.StackRight, myval, refval, dval)
			fmt.Printf("variable '%s' internal|refrence|difference\n", varname)
			fmt.Printf("%v\n", rm.ToString(spformat))
		}
		mydp.inErrorM = true
		if mydp.panicVar {
			panic("variable divergence error ...")
		}
	} else {
		//
		if  mydp.inErrorM {
			fmt.Printf("%d.%d sp '%s'[file:%s] variable '%s': returned to valid\n",
				ckp.major, ckp.minor, ckp.name, ckp.filepath, varname)
			mydp.ckp = nil
		}
		mydp.inErrorM = false
	}
}

func readCkp(name string, minor int) (vars *variableTable, ckp *checkpoint, err error) {
	err = nil
	refname := ""
	path := fmt.Sprintf("%s/%04d-%04d.%s", sppath, spmajor, minor, name)
	file, ferr := os.Open(path)
	if ferr != nil {
		vars = nil
		err = ferr
		return
	}

	refvars := make(variableTable, 20)
	linereader := bufio.NewReader(file)
	reading := true
	lineno := 0
	for reading {
		line, _, lerr := linereader.ReadLine()
		if lerr != nil {
			reading = false
			continue
		}
		if lineno == 0 {
			index := strings.Index(string(line), ":")
			refname = strings.Trim(string(line[index+1:]), " \n")
			if refname != name {
				err = errors.New(fmt.Sprintf("expecting sp: %s, found %s", name, refname))
				return
			}
		} else {
			parseVariable(string(line), &refvars)
		}
		lineno += 1
	}
	vars = &refvars
	ckp = &checkpoint{name:name, filepath:path, major:spmajor, minor:minor}
	return
}

// Parse variable line with format: <prefix>: <data>
// where <prefix> = <varname> <type> <count>
// for <type> matrix line is: <varname> matrix 1
// for <type> W the line is: <varname>.<index> W <varname-count>
// W has standard entries: d, di, dnl, dnli, r, rti, v, beta
func parseVariable(line string, vars *variableTable)  {
	index := strings.Index(line, ":")
	if index == -1 {
		// unknown line
		return
	}
	prefix := strings.Trim(line[:index], " ")
	pparts := strings.Fields(prefix)
	if len(pparts) != 3 {
		// corrupted prefix
		return
	}
	switch pparts[1] {
	case "matrix":
		// do matrix stuff
		refval, _ := matrix.FloatParseSpe(line[index+1:]) 
		(*vars)[pparts[0]] = &dataPoint{mtx:refval}
	case "epigraph":
		lpar := strings.Index(line, "[")
		rpar := strings.LastIndex(line, "]")
		mstart := strings.Index(line, "{")
		//mend   := strings.Index(line, "}")
		refval, _ := matrix.FloatParseSpe(line[mstart:rpar])
		fval, _ := strconv.ParseFloat(strings.Trim(line[lpar+1:mstart], " "), 64)
		(*vars)[pparts[0]] = &dataPoint{mtx:refval, fvar:&fval}
	case "float":
		// do variable stuff
		fval, _ := strconv.ParseFloat(strings.Trim(line[index+1:], " "), 64)
		(*vars)[pparts[0]] = &dataPoint{fvar:&fval}
	}
}
// Local Variables:
// tab-width: 4
// End:
