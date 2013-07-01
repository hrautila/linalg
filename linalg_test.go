package linalg

import (
	"fmt"
	"testing"
)

func PrintOpts(opts ...Option) {
	for i, o := range opts {
		switch o.(type) {
		case *IOpt:
			fmt.Printf("%.2d: Iopt: %s = %d\n", i, o.Name(), o.Int())
		case *FOpt:
			fmt.Printf("%.2d: Fopt: %s = %.2f\n", i, o.Name(), o.Float())
		case *SOpt:
			fmt.Printf("%.2d: Sopt: %s = '%s'\n", i, o.Name(), o.String())
		case *BOpt:
			fmt.Printf("%.2d: Bopt: %s = %v'\n", i, o.Name(), o.Bool())
		}
	}
}

func TestOpt(t *testing.T) {
	iopt := IOpt{"iopt", 10}
	fopt := FOpt{"fopt", 1.67}
	sopt := SOpt{"sopt", "value"}

	PrintOpts(&iopt, &fopt, &sopt, &BOpt{"bopt", true})
}

// Local Variables:
// tab-width: 4
// End:
