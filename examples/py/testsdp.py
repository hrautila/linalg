#
# This is copied from CVXOPT examples and modified to be used as test reference
# for corresponding Go program.
#

from cvxopt import matrix, solvers

def str2(m, fmt='%.17f'):
    s = ''
    for i in xrange(m.size[0]):
        s += "["
        for j in xrange(m.size[1]):
            if j != 0:
                s += ", "
            s += fmt % m[i, j]
        s += "]\n"
    return s


def testsdp(opts):
    c = matrix([1.,-1.,1.])  
    G = [ matrix([[-7., -11., -11., 3.],  
                  [ 7., -18., -18., 8.],  
                  [-2.,  -8.,  -8., 1.]]) ]  
    G += [ matrix([[-21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],  
                   [  0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],  
                   [ -5.,   2., -17.,   2.,  -6.,   8., -17.,  -7., 6.]]) ]  
    h = [ matrix([[33., -9.], [-9., 26.]]) ]  
    h += [ matrix([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]]) ]  

    sol = solvers.sdp(c, Gs=G, hs=h)  
    print "x = \n", str2(sol['x'], "%.9f")
    print "zs[0] = \n", str2(sol['zs'][0], "%.9f")
    print "zs[1] = \n", str2(sol['zs'][1], "%.9f")


testsdp({'maxiters': 20})

    


