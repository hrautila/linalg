#
# This is copied from CVXOPT examples and modified to be used as test reference
# for corresponding Go program.
#
# The small linear cone program of section 8.1 (Linear cone programs).


from cvxopt import matrix, solvers
from cvxopt import misc, blas

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

def testqp(opts):
    A = matrix([ [ .3, -.4,  -.2,  -.4,  1.3 ], 
                 [ .6, 1.2, -1.7,   .3,  -.3 ],
                 [-.3,  .0,   .6, -1.2, -2.0 ] ])
    b = matrix([ 1.5, .0, -1.2, -.7, .0])
    m, n = A.size

    I = matrix(0.0, (n,n))
    I[::n+1] = 1.0
    G = matrix([-I, matrix(0.0, (1,n)), I])
    h = matrix(n*[0.0] + [1.0] + n*[0.0])
    dims = {'l': n, 'q': [n+1], 's': []}
    P = A.T*A
    q = -A.T*b

    print "P=\n", str2(P, "%.17f")
    print "q=\n", str2(q, "%.17f")

    solvers.options.update(opts)
    sol = solvers.coneqp(P, q, G, h, dims, kktsolver='ldl')
    if sol['status'] == 'optimal':
        print "x=\n", str2(sol['x'])
        print "s=\n", str(sol['s'])
        print "z=\n", str2(sol['z'])

testqp({'maxiters': 10})

    
