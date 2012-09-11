#
# This is copied from CVXOPT examples and modified to be used as test reference
# for corresponding Go program.
#

from cvxopt import matrix, solvers
import helpers
import localcones


def testsdp(opts, chkpoints):
    c = matrix([1.,-1.,1.])  
    G = [ matrix([[-7., -11., -11., 3.],  
                  [ 7., -18., -18., 8.],  
                  [-2.,  -8.,  -8., 1.]]) ]  
    G += [ matrix([[-21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],  
                   [  0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],  
                   [ -5.,   2., -17.,   2.,  -6.,   8., -17.,  -7., 6.]]) ]  
    h = [ matrix([[33., -9.], [-9., 26.]]) ]  
    h += [ matrix([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]]) ]  

    if chkpoints:
        helpers.sp_reset("./sp.testsdp")
        helpers.sp_activate()
    localcones.options.update(opts)
    sol = localcones.sdp(c, Gs=G, hs=h)  
    print "x = \n", helpers.str2(sol['x'], "%.9f")
    print "zs[0] = \n", helpers.str2(sol['zs'][0], "%.9f")
    print "zs[1] = \n", helpers.str2(sol['zs'][1], "%.9f")
    print "\n *** running GO test ***"
    rungo(sol)

def rungo(sol):
    helpers.run_go_test("../testsdp", {'x': sol['x'],
                                       'ss0': sol['ss'][0],
                                       'ss1': sol['ss'][1],
                                       'zs0': sol['zs'][0],
                                       'zs1': sol['zs'][1]})

chkpoints = False
if len(sys.argv[1:]) > 0:
    if sys.argv[1] == "-sp":
        chkpoints = True

testsdp({'maxiters': 20}, chkpoints)

    


