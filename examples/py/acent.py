#
# This is copied from CVXOPT examples and modified to be used as test reference
# for corresponding Go program.
#
# The analytic centering example at the end of chapter 4 (The LAPACK 
# interface).

from cvxopt import matrix, log, mul, div, blas, lapack, base
from math import sqrt

def acent(A,b):
    """  
    Computes analytic center of A*x <= b with A m by n of rank n. 
    We assume that b > 0 and the feasible set is bounded.
    """

    MAXITERS = 100
    ALPHA = 0.01
    BETA = 0.5
    TOL = 1e-8

    ntdecrs = []
    m, n = A.size
    x = matrix(0.0, (n,1))
    H = matrix(0.0, (n,n))

    for iter in range(MAXITERS):
        
        # Gradient is g = A^T * (1./(b-A*x)).
        d = (b-A*x)**-1
        g = A.T * d

        # Hessian is H = A^T * diag(1./(b-A*x))^2 * A.
        Asc = mul( d[:,n*[0]], A)
        blas.syrk(Asc, H, trans='T')

        # Newton step is v = H^-1 * g.
        v = -g
        lapack.posv(H, v)

        # Directional derivative and Newton decrement.
        lam = blas.dot(g, v)
        ntdecrs += [ sqrt(-lam) ]
        print("%2d.  Newton decr. = %3.3e" %(iter,ntdecrs[-1]))
        if ntdecrs[-1] < TOL: return x, ntdecrs

        # Backtracking line search.
        y = mul(A*v, d)
        step = 1.0
        while 1-step*max(y) < 0: step *= BETA 
        while True:
            if -sum(log(1-step*y)) < ALPHA*step*lam: break
            step *= BETA
        x += step*v


def main(args):
    # Generate an analytic centering problem  
    #
    #    -b1 <=  Ar*x <= b2 
    #
    # with random mxn Ar and random b1, b2.

    if args[0] == "reftest":
        m, n = 10, 5
        A0 = matrix([-7.44e-01  1.11e-01  1.29e+00  2.62e+00 -1.82e+00]
                    [ 4.59e-01  7.06e-01  3.16e-01 -1.06e-01  7.80e-01]
                    [-2.95e-02 -2.22e-01 -2.07e-01 -9.11e-01 -3.92e-01]
                    [-7.75e-01  1.03e-01 -1.22e+00 -5.74e-01 -3.32e-01]
                    [-1.80e+00  1.24e+00 -2.61e+00 -9.31e-01 -6.38e-01])
        A = matrix([A0, -A0])
        b = matrix([ 8.38e-01]
                   [ 9.92e-01]
                   [ 9.56e-01]
                   [ 6.14e-01]
                   [ 6.56e-01]
                   [ 3.57e-01]
                   [ 6.36e-01]
                   [ 5.08e-01]
                   [ 8.81e-03]
                   [ 7.08e-02])
    else:
        m, n  = 500, 500
        Ar = base.normal(m,n);
        A = matrix([Ar, -Ar])
        b = base.uniform(2*m,1)

    x, ntdecrs = acent(A, b)  
    print "solution: ", str(x)
    print "ntdecrs : ", ntdecrs

def show(x, ntdecrs):
    try: 
        import pylab
    except ImportError: 
        pass
    else:
        pylab.semilogy(range(len(ntdecrs)), ntdecrs, 'o', 
                       range(len(ntdecrs)), ntdecrs, '-')
        pylab.xlabel('Iteration number')
        pylab.ylabel('Newton decrement')
        pylab.show()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
