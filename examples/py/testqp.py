
from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot 
from cvxopt.solvers import qp, options 

S = matrix( [[ 4e-2,  6e-3, -4e-3,   0.0 ], 
             [ 6e-3,  1e-2,  0.0,    0.0 ],
             [-4e-3,  0.0,   2.5e-3, 0.0 ],
             [ 0.0,   0.0,   0.0,    0.0 ]] )
pbar = matrix([.12, .10, .07, .03])


def allocation(mu, opts={}):
    n = 4
    G = matrix(0.0, (n,n))
    G[::n+1] = -1.0
    h = matrix(0.0, (n,1))
    A = matrix(1.0, (1,n))
    b = matrix(1.0)
    if opts:
        options.update(opts)

    print "mu*S=\n", mu*S
    print "-pbar=\n", -pbar
    x = qp(mu*S, -pbar, G, h, A, b)['x']
    ret = dot(pbar,x)
    risk = sqrt(dot(x, S*x))
    return x, ret, risk

def portfolio(N, opts):
    mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    xs = []
    returns = []
    risks = []
    for mu in mus:
        x, ret, rsk = allocation(mu, opts=opts)


def testone(mu, opts={}):
    x, ret, risk = allocation(mu, opts=opts)
    print "ret=%.3f, risk=%.3f" % (ret, risk)
    print "x=\n", x

testone(1.0, {'show_progress': True })

