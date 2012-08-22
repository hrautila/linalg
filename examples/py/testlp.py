
from cvxopt import matrix, solvers  
import helpers

def testlp(opts):
    c = matrix([-4., -5.])  
    G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])  
    h = matrix([3., 3., 0., 0.])  
    solvers.options.update(opts)
    sol = solvers.lp(c, G, h)  
    print"x = \n", helpers.str2(sol['x'], "%.9f")
    print"s = \n", helpers.str2(sol['s'], "%.9f")
    print"z = \n", helpers.str2(sol['z'], "%.9f")
    print "\n *** running GO test ***"
    helpers.run_go_test("../testlp", {'x': sol['x'], 's': sol['s'], 'z': sol['z']})


testlp({'solver': 'ldl'})

