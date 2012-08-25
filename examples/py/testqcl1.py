# The quadratically constrained 1-norm minimization example of section 8.7
# (Exploiting structure).

from cvxopt import blas, lapack, solvers, matrix, mul, div, setseed, normal
from math import sqrt
import helpers

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
    #print "c=\n", helpers.str2(c, "%.5f")
    #print "h=\n", helpers.str2(h, "%.5f")

    def G(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):    
        #print "Gf:x=\n", helpers.str2(x, "%.5f")
        #print "Gf:y=\n", helpers.str2(y, "%.5f")
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
        #print "end Gf:x=\n", helpers.str2(x, "%.5f")
        #print "end Gf:y=\n", helpers.str2(y, "%.5f")


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
 
        beta, v = W['beta'][0], W['v'][0]
        #print "v[1:].T =\n", helpers.str2(v[1:].T, "%.5f")
        #print "v[1:].T*A=\n", helpers.str2(v[1:].T*A, "%.5f")
        As = 2 * v * (v[1:].T * A)
        #print "As=\n", helpers.str2(As, "%.5f")
        As[1:,:] *= -1.0
        As[1:,:] -= A
        As /= beta
      
        # S = As'*As + 4 * (W1**2 + W2**2)**-1
        S = As.T * As 
        #print "S=\n", helpers.str2(S, "%.5f")
        d1, d2 = W['d'][:n], W['d'][n:]       
        print "d1=\n", helpers.str2(d1, "%.17f")
        print "d2=\n", helpers.str2(d2, "%.17f")
        d = 4.0 * (d1**2 + d2**2)**-1
        #print "d=\n", helpers.str2(d, "%.5f")
        S[::n+1] += d
        #print "S=\n", helpers.str2(S, "%.5f")
        lapack.potrf(S)
        #print "potrf S=\n", helpers.str2(S, "%.5f")

        def f(x, y, z):

            #print "f start: x=\n", helpers.str2(x, "%.5f")
            #print "f start: z=\n", helpers.str2(z, "%.5f")

            # z := - W**-T * z 
            z[:n] = -div( z[:n], d1 )
            z[n:2*n] = -div( z[n:2*n], d2 )
            z[2*n:] -= 2.0*v*( v[0]*z[2*n] - blas.dot(v[1:], z[2*n+1:]) ) 
            z[2*n+1:] *= -1.0
            z[2*n:] /= beta
            #print "f 0: z=\n", helpers.str2(z, "%.5f")

            # x := x - G' * W**-1 * z
            x[:n] -= div(z[:n], d1) - div(z[n:2*n], d2) + As.T * z[-(m+1):]
            x[n:] += div(z[:n], d1) + div(z[n:2*n], d2) 

            # Solve for x[:n]:
            #
            #    S*x[:n] = x[:n] - (W1**2 - W2**2)(W1**2 + W2**2)^-1 * x[n:]
            
            x[:n] -= mul( div(d1**2 - d2**2, d1**2 + d2**2), x[n:]) 
            #print "f potrs: x=\n", helpers.str2(x, "%.5f")
            lapack.potrs(S, x)
            
            # Solve for x[n:]:
            #
            #    (d1**-2 + d2**-2) * x[n:] = x[n:] + (d1**-2 - d2**-2)*x[:n]
             
            x[n:] += mul( d1**-2 - d2**-2, x[:n])
            x[n:] = div( x[n:], d1**-2 + d2**-2)

            # z := z + W^-T * G*x 
            z[:n] += div( x[:n] - x[n:2*n], d1) 
            z[n:2*n] += div( -x[:n] - x[n:2*n], d2) 
            z[2*n:] += As*x[:n]
            print "f end: x=\n", helpers.str2(x, "%.17f")
            print "f end: z=\n", helpers.str2(z, "%.17f")

        return f

    dims = {'l': 2*n, 'q': [m+1], 's': []}
    solvers.options['maxiters'] = 30
    sol = solvers.conelp(c, G, h, dims, kktsolver = Fkkt)
    if sol['status'] == 'optimal':
        return sol['x'][:n],  sol['z'][-m:]
    else:
        return None, None

setseed()
#m, n = 100, 100
m, n = 5, 5
A, b = normal(m,n), normal(m,1)

A = matrix([[0.66438870630377256, 1.68511096852776343, -1.47728250254375526, 0.30317355325876538, -0.89916397951294613],
            [0.83465996542735588, 0.55877932252879847, -1.06626707857638992, 1.16931080498876594, -0.56601175168881845],
            [0.32693688563254980, -0.77989544839110070, 0.10934309320941947, -1.86725147718547602, 1.55493765723389710],
            [-0.43138937120640264, 0.20898065620849879, -0.59006087009136965, -0.04384982450250739, 0.27861225756921282],
            [0.44000590962830038, 0.09061011469006654, 0.09036863350603415, 0.02113202375617339, 0.39620504458741246]])

b = matrix([-0.39126297858919096, -0.62890266671369610, -0.74060474150487765, -0.00313362362240900, -0.08087134031555188])

#print "-A", helpers.strSpe(A)
#print "-b", helpers.strSpe(b)
#print "A\n", helpers.str2(A, "%.17f", False)
#print "b\n", helpers.str2(b, "%.17f", False)

x, z = qcl1(A, b)
print "x\n", helpers.str2(x, "%.9f", False)
print "z\n", helpers.str2(z, "%.9f", False)
if x is None: print("infeasible")
