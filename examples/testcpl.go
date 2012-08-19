
package main


import (
	"github.com/hrautila/go.opt/matrix"
	"github.com/hrautila/go.opt/cvx"
	"fmt"
)

// FloorPlan implements interface cvx.ConvexProg.
type FloorPlan struct {
	Amin *matrix.FloatMatrix
}

func NewFloorPlan(plan *matrix.FloatMatrix) *FloorPlan {
	return &FloorPlan{plan}
}

func (p *FloorPlan) F0() (mnl int, x0 *matrix.FloatMatrix, err error) {
	err = nil
	mnl = 5
	x0 = matrix.FloatZeros(22, 1)
	// set last 5 elements to 1.0
	x0.Set(1.0, -1, -2, -3, -4, -5)
	return 
}

func (p *FloorPlan) F1(x *matrix.FloatMatrix)(f, Df *matrix.FloatMatrix, err error) {
	err = nil
	mn := x.Min(-1, -2, -3, -4, -5)
	if mn <= 0.0 {
		f, Df = nil, nil
		return
	}
	zeros := matrix.FloatZeros(5, 12)
	dk1 := matrix.FloatDiagonal(5, -1.0)
	dk2 := matrix.FloatZeros(5, 5)
	x17 := matrix.FloatVector(x.FloatArray()[17:])
	// -( Amin ./ (x17 .* x17) )
	diag := p.Amin.Div(x17.Mul(x17)).Scale(-1.0)
	dk2.SetIndexes(matrix.MakeDiagonalSet(5, 5), diag.FloatArray())
	Df, _ = matrix.FloatMatrixCombined(matrix.StackRight, zeros, dk1, dk2)

	x12 := matrix.FloatVector(x.FloatArray()[12:17])
	// f = -x[12:17] + div(Amin, x[17:])
	f = p.Amin.Div(x17).Minus(x12)
	return 
}

func (p *FloorPlan) F2(x, z *matrix.FloatMatrix)(f, Df, H *matrix.FloatMatrix, err error) {
	f, Df, err = p.F1(x)
	x17 := matrix.FloatVector(x.FloatArray()[17:])
	tmp := p.Amin.Div(x17.Pow(3.0))
	tmp = z.Mul(tmp).Scale(2.0)
	diag := matrix.FloatDiagonal(5, tmp.FloatArray()...)
	H = matrix.FloatZeros(22, 22)
	H.SetSubMatrix(17, 17, diag)
	return 
}

func floorplan(Amin *matrix.FloatMatrix) {

	rho := 1.0
	gamma := 5.0

	c := matrix.FloatZeros(22, 1)
	c.SetAtColumnArray(0, []int{0, 1}, []float64{1.0, 1.0})

	G := matrix.FloatZeros(26, 22)
	h := matrix.FloatZeros(26, 1)

    // -x1 <= 0  
	G.SetAt(0, 2, -1.0)

    // -x2 <= 0   
	G.SetAt(1, 3, -1.0)
	
    // -x4 <= 0  
	G.SetAt(2, 5, -1.0)
	
    // x1 - x3 + w1 <= -rho
	G.SetAtRowArray(3, []int{2, 4, 12}, []float64{1.0, -1.0, 1.0})
	h.SetAt(3, 0, -rho)
	
    // x2 - x3 + w2 <= -rho
	G.SetAtRowArray(4, []int{3, 4, 13}, []float64{1.0, -1.0, 1.0})
	h.SetAt(4, 0, -rho)

    // x3 - x5 + w3 <= -rho
	G.SetAtRowArray(5, []int{4, 6, 14}, []float64{1.0, -1.0, 1.0})
	h.SetAt(5, 0, -rho)

    // x4 - x5 + w4 <= -rho
	G.SetAtRowArray(6, []int{5, 6, 15}, []float64{1.0, -1.0, 1.0})
	h.SetAt(6, 0, -rho)

    // -W + x5 + w5 <= 0  
	G.SetAtRowArray(7, []int{0, 6, 16}, []float64{-1.0, 1.0, 1.0})

    // -y2 <= 0  
	G.SetAt(8, 8, -1.0)

    // -y3 <= 0  
	G.SetAt(9, 9, -1.0)

    // -y5 <= 0  
	G.SetAt(10, 11, -1.0)

    // -y1 + y2 + h2 <= -rho  
	G.SetAtRowArray(11, []int{7, 8, 18}, []float64{-1.0, 1.0, 1.0})
	h.SetAt(11, 0, -rho)

    // y1 - y4 + h1 <= -rho  
	G.SetAtRowArray(12, []int{7, 10, 17}, []float64{1.0, -1.0, 1.0})
	h.SetAt(12, 0, -rho)

    // y3 - y4 + h3 <= -rho  
	G.SetAtRowArray(13, []int{9, 10, 19}, []float64{1.0, -1.0, 1.0})
	h.SetAt(13, 0, -rho)

    // -H + y4 + h4 <= 0  
	G.SetAtRowArray(14, []int{1, 10, 20}, []float64{-1.0, 1.0, 1.0})

    // -H + y5 + h5 <= 0  
	G.SetAtRowArray(15, []int{1, 11, 21}, []float64{-1.0, 1.0, 1.0})

    // -w1 + h1/gamma <= 0  
	G.SetAtRowArray(16, []int{12, 17}, []float64{-1.0, 1.0/gamma})

    // w1 - gamma * h1 <= 0  
	G.SetAtRowArray(17, []int{12, 17}, []float64{1.0, -gamma})

    // -w2 + h2/gamma <= 0  
	G.SetAtRowArray(18, []int{13, 18}, []float64{-1.0, 1.0/gamma})

    //  w2 - gamma * h2 <= 0  
	G.SetAtRowArray(19, []int{13, 18}, []float64{1.0, -gamma})

    // -w3 + h3/gamma <= 0  
	G.SetAtRowArray(20, []int{14, 18}, []float64{-1.0, 1.0/gamma})

    //  w3 - gamma * h3 <= 0  
	G.SetAtRowArray(21, []int{14, 19}, []float64{1.0, -gamma})

    // -w4  + h4/gamma <= 0  
	G.SetAtRowArray(22, []int{15, 19}, []float64{-1.0, 1.0/gamma})

    //  w4 - gamma * h4 <= 0  
	G.SetAtRowArray(23, []int{15, 20}, []float64{1.0, -gamma})

    // -w5 + h5/gamma <= 0  
	G.SetAtRowArray(24, []int{16, 21}, []float64{-1.0, 1.0/gamma})

    //  w5 - gamma * h5 <= 0.0  
	G.SetAtRowArray(25, []int{16, 21}, []float64{1.0, -gamma})

	F := NewFloorPlan(Amin)

	
	var dims *cvx.DimensionSet = nil
	var solopts cvx.SolverOptions
	solopts.MaxIter = 50
	solopts.ShowProgress = true

	sol, err := cvx.Cpl(F, c, G, h, nil, nil, dims, &solopts)
	if err == nil && sol.Status == cvx.Optimal {
		x := sol.Result.At("x")[0]
		W := x.Get(0)
		H := x.Get(1)
		xs := matrix.FloatVector(x.FloatArray()[2:7])
		ys := matrix.FloatVector(x.FloatArray()[7:12])
		ws := matrix.FloatVector(x.FloatArray()[12:17])
		hs := matrix.FloatVector(x.FloatArray()[17:])
		fmt.Printf("W = %.5f, H = %.5f\n", W, H)
		fmt.Printf("x = \n%v\n", xs.ToString("%.5f"))
		fmt.Printf("y = \n%v\n", ys.ToString("%.5f"))
		fmt.Printf("w = \n%v\n", ws.ToString("%.5f"))
		fmt.Printf("h = \n%v\n", hs.ToString("%.5f"))
	} else {
		fmt.Printf("result: %v\n", err)
	}
}

func main() {
	floorplan(matrix.FloatWithValue(5, 1, 100.0))
}

// Local Variables:
// tab-width: 4
// End:
