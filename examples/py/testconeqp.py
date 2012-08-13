#
# This is copied from CVXOPT examples and modified to be used as test reference
# for corresponding Go program.
#
# The small linear cone program of section 8.1 (Linear cone programs).


from cvxopt import matrix, solvers
from cvxopt import misc, blas
import localmisc
import localcones 

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
    sol = localcones.coneqp(P, q, G, h, dims, kktsolver='ldl', options=opts)
    if sol['status'] == 'optimal':
        print "x=\n", localmisc.strMat(sol['x'])
        print "s=\n", localmisc.strMat(sol['s'])
        print "z=\n", localmisc.strMat(sol['z'])

testqp({'maxiters': 10})

    
