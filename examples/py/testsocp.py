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


def testsocp(opts):
    c = matrix([-2., 1., 5.])  

    G  = [matrix( [[12., 13.,  12.],
                   [ 6., -3., -12.],
                   [-5., -5.,  6.]] ) ]  

    G += [matrix( [[ 3.,  3., -1.,  1.],
                   [-6., -6., -9., 19.],
                   [10., -2., -2., -3.]] ) ]  

    h = [ matrix( [-12., -3., -2.] ),
          matrix( [27., 0., 3., -42.] ) ]  

    solvers.options.update(opts)
    sol = solvers.socp(c, Gq = G, hq = h)  
    
    print "x = \n", str2(sol['x'], "%.9f")
    print "zq[0] = \n", str2(sol['zq'][0], "%.9f")
    print "zq[1] = \n", str2(sol['zq'][1], "%.9f")


testsocp({'maxiters': 10})

    
