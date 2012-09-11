# The quadratically constrained 1-norm minimization example of section 8.7
# (Exploiting structure).

import sys
from cvxopt import blas, lapack, solvers, matrix, mul, div, setseed, normal
from math import sqrt
import helpers
import localcones

# helper variables for checkpointing
loopg = 0
loopf = 0

def qcl1(A, b):

    """
    Returns the solution u, z of

        (primal)  minimize    || u ||_1       
                  subject to  || A * u - b ||_2  <= 1

        (dual)    maximize    b^T z - ||z||_2
                  subject to  || A'*z ||_inf <= 1.

    Exploits structure, assuming A is m by n with m >= n. 
    """

    m, n = A.size

    # Solve equivalent cone LP with variables x = [u; v]:
    #
    #     minimize    [0; 1]' * x 
    #     subject to  [ I  -I ] * x <=  [  0 ]   (componentwise)
    #                 [-I  -I ] * x <=  [  0 ]   (componentwise)
    #                 [ 0   0 ] * x <=  [  1 ]   (SOC)
    #                 [-A   0 ]         [ -b ].
    #
    #     maximize    -t + b' * w
    #     subject to  z1 - z2 = A'*w
    #                 z1 + z2 = 1
    #                 z1 >= 0,  z2 >=0,  ||w||_2 <= t.
     
    c = matrix(n*[0.0] + n*[1.0])
    h = matrix( 0.0, (2*n + m + 1, 1))
    h[2*n] = 1.0
    h[2*n+1:] = -b

    def G(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):    
        minor = 0
        if not helpers.sp_minor_empty():
            minor = helpers.sp_minor_top()
        else:
            global loopg
            loopg += 1
            minor = loopg
        helpers.sp_create("00-Gfunc", minor)

        y *= beta
        if trans=='N':
            # y += alpha * G * x 
            y[:n] += alpha * (x[:n] - x[n:2*n]) 
            y[n:2*n] += alpha * (-x[:n] - x[n:2*n]) 
            y[2*n+1:] -= alpha * A*x[:n] 

        else:
            # y += alpha * G'*x 
            y[:n] += alpha * (x[:n] - x[n:2*n] - A.T * x[-m:])  
            y[n:] -= alpha * (x[:n] + x[n:2*n]) 

        helpers.sp_create("10-Gfunc", minor)


    def Fkkt(W): 

        # Returns a function f(x, y, z) that solves
        #
        #     [ 0   G'   ] [ x ] = [ bx ]
        #     [ G  -W'*W ] [ z ]   [ bz ].

        # First factor 
        #
        #     S = G' * W**-1 * W**-T * G
        #       = [0; -A]' * W3^-2 * [0; -A] + 4 * (W1**2 + W2**2)**-1 
        #
        # where
        #
        #     W1 = diag(d1) with d1 = W['d'][:n] = 1 ./ W['di'][:n]  
        #     W2 = diag(d2) with d2 = W['d'][n:] = 1 ./ W['di'][n:]  
        #     W3 = beta * (2*v*v' - J),  W3^-1 = 1/beta * (2*J*v*v'*J - J)  
        #        with beta = W['beta'][0], v = W['v'][0], J = [1, 0; 0, -I].
  
        # As = W3^-1 * [ 0 ; -A ] = 1/beta * ( 2*J*v * v' - I ) * [0; A]
 
        minor = 0
        if not helpers.sp_minor_empty():
            minor = helpers.sp_minor_top()

        beta, v = W['beta'][0], W['v'][0]
        As = 2 * v * (v[1:].T * A)
        As[1:,:] *= -1.0
        As[1:,:] -= A
        As /= beta
      
        # S = As'*As + 4 * (W1**2 + W2**2)**-1
        S = As.T * As 
        d1, d2 = W['d'][:n], W['d'][n:]       
        d = 4.0 * (d1**2 + d2**2)**-1
        S[::n+1] += d
        lapack.potrf(S)

        def f(x, y, z):

            minor = 0
            if not helpers.sp_minor_empty():
                minor = helpers.sp_minor_top()
            else:
                global loopf
                loopf += 1
                minor = loopf
            helpers.sp_create("00-f", minor)
  
            # z := - W**-T * z 
            z[:n] = -div( z[:n], d1 )
            z[n:2*n] = -div( z[n:2*n], d2 )

            z[2*n:] -= 2.0*v*( v[0]*z[2*n] - blas.dot(v[1:], z[2*n+1:]) ) 
            z[2*n+1:] *= -1.0
            z[2*n:] /= beta

              # x := x - G' * W**-1 * z
            x[:n] -= div(z[:n], d1) - div(z[n:2*n], d2) + As.T * z[-(m+1):]
            x[n:] += div(z[:n], d1) + div(z[n:2*n], d2) 
            helpers.sp_create("15-f", minor)

  
            # Solve for x[:n]:
            #
            #    S*x[:n] = x[:n] - (W1**2 - W2**2)(W1**2 + W2**2)^-1 * x[n:]
            
            x[:n] -= mul( div(d1**2 - d2**2, d1**2 + d2**2), x[n:]) 
            helpers.sp_create("25-f", minor)

            lapack.potrs(S, x)
            helpers.sp_create("30-f", minor)
            
            # Solve for x[n:]:
            #
            #    (d1**-2 + d2**-2) * x[n:] = x[n:] + (d1**-2 - d2**-2)*x[:n]
             
            x[n:] += mul( d1**-2 - d2**-2, x[:n])
            helpers.sp_create("35-f", minor)

            x[n:] = div( x[n:], d1**-2 + d2**-2)
            helpers.sp_create("40-f", minor)

            # z := z + W^-T * G*x 
            z[:n] += div( x[:n] - x[n:2*n], d1) 
            helpers.sp_create("44-f", minor)

            z[n:2*n] += div( -x[:n] - x[n:2*n], d2) 
            helpers.sp_create("48-f", minor)

            z[2*n:] += As*x[:n]
            helpers.sp_create("50-f", minor)
  
        return f

    dims = {'l': 2*n, 'q': [m+1], 's': []}
    localcones.options['maxiters'] = 30
    sol = localcones.conelp(c, G, h, dims, kktsolver = Fkkt)
    if sol['status'] == 'optimal':
        return sol['x'][:n],  sol['z'][-m:]
    else:
        return None, None

def rungo(A, b, x, z):
    helpers.run_go_test("../testqcl1", {'x': x, 'z': z, 'A': A, 'b': b})

setseed()
#m, n = 100, 100
m, n = 10, 10
A, b = normal(m,n), normal(m,1)

no_go = False
if len(sys.argv[1:]) > 0 and sys.argv[1] == '-sp':
    helpers.sp_reset("./sp.qcl1")
    helpers.sp_activate()
    no_go = True

x, z = qcl1(A, b)
if x is None:
    print("infeasible")
    x = matrix(0.0, (0, 1))
    z = matrix(0.0, (0, 1))
else:
    print "x\n", helpers.str2(x, "%.9f")
    print "z\n", helpers.str2(z, "%.9f")

if not no_go:
    rungo(A, b, x, z)
else:
    print "A=\"%s\"\n" % helpers.strSpe(A)
    print "b=\"%s\"\n" % helpers.strSpe(b)
