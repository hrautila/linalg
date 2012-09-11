
from cvxopt import matrix, blas, lapack, misc, base

import helpers
import math
import StringIO

def strM(m):
    s = ''
    for i in xrange(len(m)):
        if i != 0:
            s += ', '
        s += "%.17f" % m[i]

    return "{%d %d [%s]}" % (m.size[0], m.size[1], s)


def str2(m, fmt):
    s = ''
    for i in xrange(m.size[0]):
        s += "["
        for j in xrange(m.size[1]):
            if j != 0:
                s += ", "
            s += fmt % m[i, j]
        s += "]\n"
    return s

def strMat(m):
    s = ''
    for i in xrange(len(m)):
        if i != 0:
            s += '\n'
        s += "[%.17f]" % m[i]

    return s


def printWData(W):
    for k in W.keys():
        if k == 'beta':
            print "beta: \"{%d 1 %s}\"" % (len(W[k]), W[k])
        elif isinstance(W[k], list):
            for n in range(len(W[k])):
                print "%s[%d]: \"%s\"" %(k, n, strM(W[k][n]))
        else:
            print "%s[0]: \"%s\"" % (k, strM(W[k]))

def printW(W):
    for k in W.keys():
        if k == 'beta':
            print "** beta **\n", W[k]
        elif isinstance(W[k], list):
            for n in range(len(W[k])):
                print "** %s[%d] **\n" %(k, n), strMat(W[k][n])
        else:
            print "** %s[0] **\n" % k, strMat(W[k])

def local_pack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0):
     """
     Copy x to y using packed storage.
    
     The vector x is an element of S, with the 's' components stored in 
     unpacked storage.  On return, x is copied to y with the 's' components
     stored in packed storage and the off-diagonal entries scaled by 
     sqrt(2).
     """

     nlq = mnl + dims['l'] + sum(dims['q'])
     blas.copy(x, y, n = nlq, offsetx = offsetx, offsety = offsety)
     iu, ip = offsetx + nlq, offsety + nlq
     for n in dims['s']:
         for k in range(n):
             blas.copy(x, y, n = n-k, offsetx = iu + k*(n+1), offsety = ip)
             y[ip] /= math.sqrt(2)
             ip += n-k
         iu += n**2 
     np = sum([ int(n*(n+1)/2) for n in dims['s'] ])
     blas.scal(math.sqrt(2.0), y, n = np, offset = offsety+nlq)


def local_unpack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0):
     """
     The vector x is an element of S, with the 's' components stored
     in unpacked storage and off-diagonal entries scaled by sqrt(2).
     On return, x is copied to y with the 's' components stored in 
     unpacked storage.
     """

     import math
     nlq = mnl + dims['l'] + sum(dims['q'])
     blas.copy(x, y, n = nlq, offsetx = offsetx, offsety = offsety)
     iu, ip = offsety+nlq, offsetx+nlq
     for n in dims['s']:
         for k in range(n):
             ##print "ip=%d,iu=%d, n=%d,k=%d, s-n=%d, s-oy=%d" % (ip,iu,n,k,n-k-1, iu+k*(n+1)+1)
             blas.copy(x, y, n = n-k, offsetx = ip, offsety = iu+k*(n+1))
             #y[iu+k*(n+1)] *= math.sqrt(2)
             blas.scal(1.0/math.sqrt(2.0), y, n = n-k-1, offset = iu+k*(n+1)+1)
             ip += n-k
         iu += n**2 
     #nu = sum([ n**2 for n in dims['s'] ])
     #blas.scal(1.0/math.sqrt(2.0), y, n = nu, offset = offsety+nlq)

def local_max_step(x, dims, mnl = 0, sigma = None):
    """
    Returns min {t | x + t*e >= 0}, where e is defined as follows
    
    - For the nonlinear and 'l' blocks: e is the vector of ones.
    - For the 'q' blocks: e is the first unit vector.
    - For the 's' blocks: e is the identity matrix.
    
    When called with the argument sigma, also returns the eigenvalues 
    (in sigma) and the eigenvectors (in x) of the 's' components of x.
    """

    t = []
    ind = mnl + dims['l']
    if ind: t += [ -min(x[:ind]) ] 
    for m in dims['q']:
        if m: t += [ blas.nrm2(x, offset = ind + 1, n = m-1) - x[ind] ]
        ind += m
    if sigma is None and dims['s']:  
        Q = matrix(0.0, (max(dims['s']), max(dims['s'])))
        w = matrix(0.0, (max(dims['s']),1))
    ind2 = 0
    for m in dims['s']:
        if sigma is None:
            blas.copy(x, Q, offsetx = ind, n = m**2)
            lapack.syevr(Q, w, range = 'I', il = 1, iu = 1, n = m, ldA = m)
            if m:  t += [ -w[0] ]
        else:            
            lapack.syevd(x, sigma, jobz = 'V', n = m, ldA = m, offsetA = 
                ind, offsetW = ind2)
            if m:  t += [ -sigma[ind2] ] 
        ind += m*m
        ind2 += m
    if t: return max(t)
    else: return 0.0


def local_sdot(x, y, dims, mnl = 0):
    """
    Inner product of two vectors in S.
    """
    
    ind = mnl + dims['l'] + sum(dims['q'])
    a = blas.dot(x, y, n = ind)
    for m in dims['s']:
        a += blas.dot(x, y, offsetx = ind, offsety = ind, incx = m+1, 
            incy = m+1, n = m)
        for j in range(1, m):
            a += 2.0 * blas.dot(x, y, incx = m+1, incy = m+1, 
                offsetx = ind+j, offsety = ind+j, n = m-j)
        ind += m**2
    return a



def local_sinv(x, y, dims, mnl = 0):   
    """
    The inverse product x := (y o\ x), when the 's' components of y are 
    diagonal.
    """
    
    # For the nonlinear and 'l' blocks:  
    # 
    #     yk o\ xk = yk .\ xk.

    blas.tbsv(y, x, n = mnl + dims['l'], k = 0, ldA = 1)


    # For the 'q' blocks: 
    #
    #                        [ l0   -l1'              ]  
    #     yk o\ xk = 1/a^2 * [                        ] * xk
    #                        [ -l1  (a*I + l1*l1')/l0 ]
    #
    # where yk = (l0, l1) and a = l0^2 - l1'*l1.

    ind = mnl + dims['l']
    for m in dims['q']:
        aa = local_jnrm2(y, n = m, offset = ind)
        print "a =", a
        aa = aa ** 2
        cc = x[ind]
        dd = blas.dot(y, x, offsetx = ind+1, offsety = ind+1, n = m-1)
        x[ind] = cc * y[ind] - dd
        blas.scal(aa / y[ind], x, n = m-1, offset = ind+1)
        blas.axpy(y, x, alpha = dd/y[ind] - cc, n = m-1, offsetx = ind+1, 
            offsety = ind+1)
        blas.scal(1.0/aa, x, n = m, offset = ind)
        ind += m


    # For the 's' blocks:
    #
    #     yk o\ xk =  xk ./ gamma
    #
    # where gammaij = .5 * (yk_i + yk_j).

    ind2 = ind
    for m in dims['s']:
        for j in range(m):
            u = 0.5 * ( y[ind2+j:ind2+m] + y[ind2+j] )
            blas.tbsv(u, x, n = m-j, k = 0, ldA = 1, offsetx = ind + 
                j*(m+1))  
        ind += m*m
        ind2 += m


def local_jnrm2(x, n = None, offset = 0):
    """
    Returns sqrt(x' * J * x) where J = [1, 0; 0, -I], for a vector
    x in a second order cone. 
    """

    if n is None:  n = len(x)
    a = blas.nrm2(x, n = n-1, offset = offset+1)
    #print "jnrm2: a =", a, " x[offset] = ", x[offset]
    return math.sqrt(x[offset] - a) * math.sqrt(x[offset] + a)


def local_sprod(x, y, dims, mnl = 0, diag = 'N'):   
    """
    The product x := (y o x).  If diag is 'D', the 's' part of y is 
    diagonal and only the diagonal is stored.
    """


    # For the nonlinear and 'l' blocks:  
    #
    #     yk o xk = yk .* xk.

    blas.tbmv(y, x, n = mnl + dims['l'], k = 0, ldA = 1) 
    #print "sprod l: x=\n", x

    # For 'q' blocks: 
    #
    #               [ l0   l1'  ]
    #     yk o xk = [           ] * xk
    #               [ l1   l0*I ] 
    #
    # where yk = (l0, l1).
    
    ind = mnl + dims['l']
    for m in dims['q']:
        dd = blas.dot(x, y, offsetx = ind, offsety = ind, n = m)
        #print "dd=", dd
        #print "scal=", y[ind]
        blas.scal(y[ind], x, offset = ind+1, n = m-1)
        #print "axpy=", x[ind]
        blas.axpy(y, x, alpha = x[ind], n = m-1, offsetx = ind+1, offsety 
            = ind+1)
        x[ind] = dd
        ind += m
    #print "sprod q: x=\n", x


    # For the 's' blocks:
    #
    #    yk o sk = .5 * ( Yk * mat(xk) + mat(xk) * Yk )
    # 
    # where Yk = mat(yk) if diag is 'N' and Yk = diag(yk) if diag is 'D'.

    if diag is 'N':
        maxm = max([0] + dims['s'])
        A = matrix(0.0, (maxm, maxm))

        for m in dims['s']:
            blas.copy(x, A, offsetx = ind, n = m*m)

            # Write upper triangular part of A and yk.
            for i in range(m-1):
                misc.symm(A, m)
                misc.symm(y, m, offset = ind)

            # xk = 0.5 * (A*yk + yk*A)
            blas.syr2k(A, y, x, alpha = 0.5, n = m, k = m, ldA = m,  ldB = 
                m, ldC = m, offsetB = ind, offsetC = ind)

            ind += m*m

        #print "sprod diag=N s: x=\n", x

    else:
        ind2 = ind
        for m in dims['s']:
            for j in range(m):
                u = 0.5 * ( y[ind2+j:ind2+m] + y[ind2+j] )
                #print "u.size=", u.size, "u=\n", u
                blas.tbmv(u, x, n = m-j, k = 0, ldA = 1, offsetx = 
                    ind + j*(m+1))  
            ind += m*m
            ind2 += m
        #print "sprod diag=T s: x=\n", x


def local_sgemv(A, x, y, dims, trans = 'N', alpha = 1.0, beta = 0.0, n = None, 
    offsetA = 0, offsetx = 0, offsety = 0): 
    """
    Matrix-vector multiplication.

    A is a matrix or spmatrix of size (m, n) where 
    
        N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ) 

    representing a mapping from R^n to S.  
    
    If trans is 'N': 
    
        y := alpha*A*x + beta * y   (trans = 'N').
    
    x is a vector of length n.  y is a vector of length N.
    
    If trans is 'T':
    
        y := alpha*A'*x + beta * y  (trans = 'T').
    
    x is a vector of length N.  y is a vector of length n.
    
    The 's' components in S are stored in unpacked 'L' storage.
    """

    m = dims['l'] + sum(dims['q']) + sum([ k**2 for k in dims['s'] ]) 
    if n is None: n = A.size[1]
    if trans == 'T' and alpha:
        misc.trisc(x, dims, offsetx)
        #print "trisc x:\n", strMat(x)
    #print "alpha=%.4f beta=%.4f m=%d, n=%d" % (alpha, beta, m, n)
    #print "A=\n", A, "\nx=\n", strMat(x), "\ny=\n", strMat(y)
    base.gemv(A, x, y, trans = trans, alpha = alpha, beta = beta, m = m,
        n = n, offsetA = offsetA, offsetx = offsetx, offsety = offsety)
    #print "gemv y:\n", strMat(y)
    if trans == 'T' and alpha:
        misc.triusc(x, dims, offsetx)


def scale(x, W, trans = 'N', inverse = 'N'):  
    """
    Applies Nesterov-Todd scaling or its inverse.
    
    Computes 
    
         x := W*x        (trans is 'N', inverse = 'N')  
         x := W^T*x      (trans is 'T', inverse = 'N')  
         x := W^{-1}*x   (trans is 'N', inverse = 'I')  
         x := W^{-T}*x   (trans is 'T', inverse = 'I'). 
    
    x is a dense 'd' matrix.
    
    W is a dictionary with entries:
    
    - W['dnl']: positive vector
    - W['dnli']: componentwise inverse of W['dnl']
    - W['d']: positive vector
    - W['di']: componentwise inverse of W['d']
    - W['v']: lists of 2nd order cone vectors with unit hyperbolic norms
    - W['beta']: list of positive numbers
    - W['r']: list of square matrices 
    - W['rti']: list of square matrices.  rti[k] is the inverse transpose
      of r[k].
    
    The 'dnl' and 'dnli' entries are optional, and only present when the 
    function is called from the nonlinear solver.
    """

    ind = 0

    minor = 0
    if not helpers.sp_minor_empty():
        minor = helpers.sp_minor_top()

    #print "%d.%04d scaling x=\n" % (helpers.sp_major(), minor), helpers.str2(x, "%.17f")

    # Scaling for nonlinear component xk is xk := dnl .* xk; inverse 
    # scaling is xk ./ dnl = dnli .* xk, where dnl = W['dnl'], 
    # dnli = W['dnli'].

    if 'dnl' in W:
        if inverse == 'N': w = W['dnl']
        else: w = W['dnli']
        for k in xrange(x.size[1]):
            blas.tbmv(w, x, n = w.size[0], k = 0, ldA = 1, offsetx = 
                k*x.size[0])
        ind += w.size[0]

    ##print "phase1: x=\n", x

    if not helpers.sp_minor_empty():
        helpers.sp_create("000scale", minor)

    # Scaling for linear 'l' component xk is xk := d .* xk; inverse 
    # scaling is xk ./ d = di .* xk, where d = W['d'], di = W['di'].

    if inverse == 'N': w = W['d']
    else: w = W['di']
    for k in xrange(x.size[1]):
        blas.tbmv(w, x, n = w.size[0], k = 0, ldA = 1, offsetx = 
            k*x.size[0] + ind)
    ind += w.size[0]
  
    ##print "phase2: x=\n", x
    if not helpers.sp_minor_empty():
        helpers.sp_create("010scale", minor)
    
    # Scaling for 'q' component is 
    #
    #     xk := beta * (2*v*v' - J) * xk
    #         = beta * (2*v*(xk'*v)' - J*xk)
    #
    # where beta = W['beta'][k], v = W['v'][k], J = [1, 0; 0, -I].
    #
    # Inverse scaling is
    #
    #     xk := 1/beta * (2*J*v*v'*J - J) * xk
    #         = 1/beta * (-J) * (2*v*((-J*xk)'*v)' + xk). 

    w = matrix(0.0, (x.size[1], 1))
    for k in xrange(len(W['v'])):
        v = W['v'][k]
        m = v.size[0]
        if inverse == 'I':  
            blas.scal(-1.0, x, offset = ind, inc = x.size[0])
        blas.gemv(x, v, w, trans = 'T', m = m, n = x.size[1], offsetA = 
            ind, ldA = x.size[0])
        blas.scal(-1.0, x, offset = ind, inc = x.size[0])
        blas.ger(v, w, x, alpha = 2.0, m = m, n = x.size[1], ldA = 
            x.size[0], offsetA = ind)
        if inverse == 'I': 
            blas.scal(-1.0, x, offset = ind, inc = x.size[0])
            a = 1.0 / W['beta'][k] 
        else:
            a = W['beta'][k] 
        for i in xrange(x.size[1]):
            blas.scal(a, x, n = m, offset = ind + i*x.size[0])
        ind += m

    ##print "phase3: x=\n", x
    if not helpers.sp_minor_empty():
        helpers.sp_create("020scale", minor)
        
    # Scaling for 's' component xk is
    #
    #     xk := vec( r' * mat(xk) * r )  if trans = 'N'
    #     xk := vec( r * mat(xk) * r' )  if trans = 'T'.
    #
    # r is kth element of W['r'].
    #
    # Inverse scaling is
    #
    #     xk := vec( rti * mat(xk) * rti' )  if trans = 'N'
    #     xk := vec( rti' * mat(xk) * rti )  if trans = 'T'.
    #
    # rti is kth element of W['rti'].

    maxn = max( [0] + [ r.size[0] for r in W['r'] ] )
    a = matrix(0.0, (maxn, maxn))
    for k in xrange(len(W['r'])):

        if inverse == 'N':
            r = W['r'][k]
            if trans == 'N': t = 'T'
            else: t = 'N'
        else:
            r = W['rti'][k]
            t = trans

        n = r.size[0]
        for i in xrange(x.size[1]):

            # scale diagonal of xk by 0.5
            blas.scal(0.5, x, offset = ind + i*x.size[0], inc = n+1, n = n)

            # a = r*tril(x) (t is 'N') or a = tril(x)*r  (t is 'T')
            blas.copy(r, a)
            if t == 'N':   
                blas.trmm(x, a, side = 'R', m = n, n = n, ldA = n, ldB = n,
                    offsetA = ind + i*x.size[0])
                ##print "N: a=\n", a
            else:    
                ##print "T pre: oa=",ind+i*x.size[0]," a=\n", a, "x=\n", x
                blas.trmm(x, a, side = 'L', m = n, n = n, ldA = n, ldB = n,
                    offsetA = ind + i*x.size[0])
                ##print "T post: a=\n", a
 
            # x := (r*a' + a*r')  if t is 'N'
            # x := (r'*a + a'*r)  if t is 'T'
            blas.syr2k(r, a, x, trans = t, n = n, k = n, ldB = n, ldC = n,
                offsetC = ind + i*x.size[0])
 
        ind += n**2

    if not helpers.sp_minor_empty():
        helpers.sp_create("030scale", minor)
    ##print "phase4: x=\n", x


def scale2(lmbda, x, dims, mnl = 0, inverse = 'N'):
    """
    Evaluates

        x := H(lambda^{1/2}) * x   (inverse is 'N')
        x := H(lambda^{-1/2}) * x  (inverse is 'I').
    
    H is the Hessian of the logarithmic barrier.
    """
      
    minor = 0
    if not helpers.sp_minor_empty():
        minor = helpers.sp_minor_top()

    #print "%d.%04d scale2 x=\n" % (helpers.sp_major(), minor), helpers.str2(x, "%.17f")
    #print "lmbda=\n", helpers.str2(lmbda, "%.17f")
    

    if not helpers.sp_minor_empty():
        helpers.sp_create("000scale2", minor)

    # For the nonlinear and 'l' blocks, 
    #
    #     xk := xk ./ l   (inverse is 'N')
    #     xk := xk .* l   (inverse is 'I')
    #
    # where l is lmbda[:mnl+dims['l']].

    if inverse == 'N':
        blas.tbsv(lmbda, x, n = mnl + dims['l'], k = 0, ldA = 1)
    else:
        blas.tbmv(lmbda, x, n = mnl + dims['l'], k = 0, ldA = 1)
   
    if not helpers.sp_minor_empty():
        helpers.sp_create("010scale2", minor)

  
    # For 'q' blocks, if inverse is 'N',
    #
    #     xk := 1/a * [ l'*J*xk;  
    #         xk[1:] - (xk[0] + l'*J*xk) / (l[0] + 1) * l[1:] ].
    #
    # If inverse is 'I',
    #
    #     xk := a * [ l'*xk; 
    #         xk[1:] + (xk[0] + l'*xk) / (l[0] + 1) * l[1:] ].
    #
    # a = sqrt(lambda_k' * J * lambda_k), l = lambda_k / a.

    ind = mnl + dims['l']
    for m in dims['q']:
        a = misc.jnrm2(lmbda, n = m, offset = ind)
        if inverse == 'N':
            lx = misc.jdot(lmbda, x, n = m, offsetx = ind, offsety = ind)/a
        else:
            lx = blas.dot(lmbda, x, n = m, offsetx = ind, offsety = ind)/a
        x0 = x[ind]
        x[ind] = lx
        c = (lx + x0) / (lmbda[ind]/a + 1) / a 
        if inverse == 'N':  c *= -1.0
        blas.axpy(lmbda, x, alpha = c, n = m-1, offsetx = ind+1, offsety =
            ind+1)
        if inverse == 'N': a = 1.0/a 
        blas.scal(a, x, offset = ind, n = m)
        ind += m

    if not helpers.sp_minor_empty():
        helpers.sp_create("020scale2", minor)

        

    # For the 's' blocks, if inverse is 'N',
    #
    #     xk := vec( diag(l)^{-1/2} * mat(xk) * diag(k)^{-1/2}).
    #
    # If inverse is 'I',
    #
    #     xk := vec( diag(l)^{1/2} * mat(xk) * diag(k)^{1/2}).
    #
    # where l is kth block of lambda.
    # 
    # We scale upper and lower triangular part of mat(xk) because the
    # inverse operation will be applied to nonsymmetric matrices.

    ind2 = ind
    for k in range(len(dims['s'])):
        m = dims['s'][k]
        for j in range(m):
            c = math.sqrt(lmbda[ind2+j]) * base.sqrt(lmbda[ind2:ind2+m])
            if inverse == 'N':  
                blas.tbsv(c, x, n = m, k = 0, ldA = 1, offsetx = ind + j*m)
            else:
                blas.tbmv(c, x, n = m, k = 0, ldA = 1, offsetx = ind + j*m)
        ind += m*m
        ind2 += m

    if not helpers.sp_minor_empty():
        helpers.sp_create("030scale2", minor)



def local_compute_scaling(s, z, lmbda, dims, mnl = None):
    """
    Returns the Nesterov-Todd scaling W at points s and z, and stores the 
    scaled variable in lmbda. 
    
        W * z = W^{-T} * s = lmbda. 

    """
     
    W = {}

    # For the nonlinear block:
    #
    #     W['dnl'] = sqrt( s[:mnl] ./ z[:mnl] )
    #     W['dnli'] = sqrt( z[:mnl] ./ s[:mnl] )
    #     lambda[:mnl] = sqrt( s[:mnl] .* z[:mnl] )

    if mnl is None:
        mnl = 0
    else:
        W['dnl'] = base.sqrt( base.div( s[:mnl], z[:mnl] ))
        W['dnli'] = W['dnl']**-1
        lmbda[:mnl] = base.sqrt( base.mul( s[:mnl], z[:mnl] ) ) 
        

    # For the 'l' block: 
    #
    #     W['d'] = sqrt( sk ./ zk )
    #     W['di'] = sqrt( zk ./ sk )
    #     lambdak = sqrt( sk .* zk )
    #
    # where sk and zk are the first dims['l'] entries of s and z.
    # lambda_k is stored in the first dims['l'] positions of lmbda.
             
    m = dims['l']
    W['d'] = base.sqrt( base.div( s[mnl:mnl+m], z[mnl:mnl+m] ))
    #print "d.size =", W['d'].size, " data=\n", W['d']
    W['di'] = W['d']**-1
    lmbda[mnl:mnl+m] = base.sqrt( base.mul( s[mnl:mnl+m], z[mnl:mnl+m] ) ) 
    #print "after l:\n", lmbda

    # For the 'q' blocks, compute lists 'v', 'beta'.
    #
    # The vector v[k] has unit hyperbolic norm: 
    # 
    #     (sqrt( v[k]' * J * v[k] ) = 1 with J = [1, 0; 0, -I]).
    # 
    # beta[k] is a positive scalar.
    #
    # The hyperbolic Householder matrix H = 2*v[k]*v[k]' - J
    # defined by v[k] satisfies 
    # 
    #     (beta[k] * H) * zk  = (beta[k] * H) \ sk = lambda_k
    #
    # where sk = s[indq[k]:indq[k+1]], zk = z[indq[k]:indq[k+1]].
    #
    # lambda_k is stored in lmbda[indq[k]:indq[k+1]].
           
    ind = mnl + dims['l']
    W['v'] = [ matrix(0.0, (k,1)) for k in dims['q'] ]
    W['beta'] = len(dims['q']) * [ 0.0 ] 

    for k in range(len(dims['q'])):
        m = dims['q'][k]
        v = W['v'][k]

        # a = sqrt( sk' * J * sk )  where J = [1, 0; 0, -I]
        aa = misc.jnrm2(s, offset = ind, n = m)
        #print "aa =", aa
        # b = sqrt( zk' * J * zk )
        bb = misc.jnrm2(z, offset = ind, n = m) 
        #print "bb =", bb

        # beta[k] = ( a / b )**1/2
        W['beta'][k] = math.sqrt( aa / bb )

        # c = sqrt( (sk/a)' * (zk/b) + 1 ) / sqrt(2)    
        cc = math.sqrt( ( blas.dot(s, z, n = m, offsetx = ind, offsety = 
            ind) / aa / bb + 1.0 ) / 2.0 )

        # vk = 1/(2*c) * ( (sk/a) + J * (zk/b) )
        blas.copy(z, v, offsetx = ind, n = m)
        blas.scal(-1.0/bb, v)
        v[0] *= -1.0 
        blas.axpy(s, v, 1.0/aa, offsetx = ind, n = m)
        blas.scal(1.0/2.0/cc, v)

        # v[k] = 1/sqrt(2*(vk0 + 1)) * ( vk + e ),  e = [1; 0]
        v[0] += 1.0
        blas.scal(1.0/math.sqrt(2.0 * v[0]), v)
            
        # To get the scaled variable lambda_k
        # 
        #     d =  sk0/a + zk0/b + 2*c
        #     lambda_k = [ c; 
        #                  (c + zk0/b)/d * sk1/a + (c + sk0/a)/d * zk1/b ]
        #     lambda_k *= sqrt(a * b)

        lmbda[ind] = cc
        dd = 2*cc + s[ind]/aa + z[ind]/bb
        blas.copy(s, lmbda, offsetx = ind+1, offsety = ind+1, n = m-1) 
        blas.scal((cc + z[ind]/bb)/dd/aa, lmbda, n = m-1, offset = ind+1)
        blas.axpy(z, lmbda, (cc + s[ind]/aa)/dd/bb, n = m-1, offsetx = 
            ind+1, offsety = ind+1)
        blas.scal(math.sqrt(aa*bb), lmbda, offset = ind, n = m)

        ind += m
        #print "W['v'][%d].size =" % k, W['v'][k].size, " data=\n", W['v'][k]
        #print "W['beta'][%d] =" % k, W['beta'][k]
        #print "after q[%d]:\n" % k, lmbda


    # For the 's' blocks: compute two lists 'r' and 'rti'.
    #
    #     r[k]' * sk^{-1} * r[k] = diag(lambda_k)^{-1}
    #     r[k]' * zk * r[k] = diag(lambda_k)
    #
    # where sk and zk are the entries inds[k] : inds[k+1] of
    # s and z, reshaped into symmetric matrices.
    #
    # rti[k] is the inverse of r[k]', so 
    #
    #     rti[k]' * sk * rti[k] = diag(lambda_k)^{-1}
    #     rti[k]' * zk^{-1} * rti[k] = diag(lambda_k).
    #
    # The vectors lambda_k are stored in 
    # 
    #     lmbda[ dims['l'] + sum(dims['q']) : -1 ]
            
    W['r'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
    W['rti'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
    work = matrix(0.0, (max( [0] + dims['s'] )**2, 1))
    Ls = matrix(0.0, (max( [0] + dims['s'] )**2, 1))
    Lz = matrix(0.0, (max( [0] + dims['s'] )**2, 1))

    ind2 = ind
    for k in range(len(dims['s'])):
        m = dims['s'][k]
        r, rti = W['r'][k], W['rti'][k]

        # Factor sk = Ls*Ls'; store Ls in ds[inds[k]:inds[k+1]].
        blas.copy(s, Ls, offsetx = ind2, n = m**2) 
        lapack.potrf(Ls, n = m, ldA = m)

        # Factor zs[k] = Lz*Lz'; store Lz in dz[inds[k]:inds[k+1]].
        blas.copy(z, Lz, offsetx = ind2, n = m**2) 
        lapack.potrf(Lz, n = m, ldA = m)
	 
        # SVD Lz'*Ls = U*diag(lambda_k)*V'.  Keep U in work. 
        for i in range(m): 
            blas.scal(0.0, Ls, offset = i*m, n = i)
        blas.copy(Ls, work, n = m**2)
        blas.trmm(Lz, work, transA = 'T', ldA = m, ldB = m, n = m, m = m) 
        lapack.gesvd(work, lmbda, jobu = 'O', ldA = m, m = m, n = m, 
            offsetS = ind)
	       
        # r = Lz^{-T} * U 
        blas.copy(work, r, n = m*m)
        blas.trsm(Lz, r, transA = 'T', m = m, n = m, ldA = m)

        # rti = Lz * U 
        blas.copy(work, rti, n = m*m)
        blas.trmm(Lz, rti, m = m, n = m, ldA = m)

        # r := r * diag(sqrt(lambda_k))
        # rti := rti * diag(1 ./ sqrt(lambda_k))
        for i in range(m):
            a = math.sqrt( lmbda[ind+i] )
            blas.scal(a, r, offset = m*i, n = m)
            blas.scal(1.0/a, rti, offset = m*i, n = m)

        ind += m
        ind2 += m*m

    return W




def local_update_scaling(W, lmbda, s, z):
    """
    Updates the Nesterov-Todd scaling matrix W and the scaled variable 
    lmbda so that on exit
    
          W * zt = W^{-T} * st = lmbda.
     
    On entry, the nonlinear, 'l' and 'q' components of the arguments s 
    and z contain W^{-T}*st and W*zt, i.e, the new iterates in the current 
    scaling.
    
    The 's' components contain the factors Ls, Lz in a factorization of 
    the new iterates in the current scaling, W^{-T}*st = Ls*Ls',   
    W*zt = Lz*Lz'.
    """
  

    # Nonlinear and 'l' blocks
    #
    #    d :=  d .* sqrt( s ./ z )
    #    lmbda := lmbda .* sqrt(s) .* sqrt(z)

    if 'dnl' in W:
        mnl = len(W['dnl'])
    else:
        mnl = 0
    ml = len(W['d'])
    m = mnl + ml
    #print "ml=%d, mnl=%d, m=%d" % (ml, mnl, m)
    s[:m] = base.sqrt( s[:m] )
    z[:m] = base.sqrt( z[:m] )
 
    # d := d .* s .* z 
    if 'dnl' in W:
        blas.tbmv(s, W['dnl'], n = mnl, k = 0, ldA = 1)
        blas.tbsv(z, W['dnl'], n = mnl, k = 0, ldA = 1)
        W['dnli'][:] = W['dnl'][:] ** -1
    blas.tbmv(s, W['d'], n = ml, k = 0, ldA = 1, offsetA = mnl)
    blas.tbsv(z, W['d'], n = ml, k = 0, ldA = 1, offsetA = mnl)
    W['di'][:] = W['d'][:] ** -1
         
    # lmbda := s .* z
    blas.copy(s, lmbda, n = m)
    blas.tbmv(z, lmbda, n = m, k = 0, ldA = 1)
    #print "-- end of l:\nz=\n", strMat(z), "\nlmbda=\n", strMat(lmbda)
    #print "W[d] =\n", strMat(W['d'])
    #print "W[di] =\n", strMat(W['di'])

    # 'q' blocks.
    # 
    # Let st and zt be the new variables in the old scaling:
    #
    #     st = s_k,   zt = z_k
    #
    # and a = sqrt(st' * J * st),  b = sqrt(zt' * J * zt).
    #
    # 1. Compute the hyperbolic Householder transformation 2*q*q' - J 
    #    that maps st/a to zt/b.
    # 
    #        c = sqrt( (1 + st'*zt/(a*b)) / 2 ) 
    #        q = (st/a + J*zt/b) / (2*c). 
    #
    #    The new scaling point is 
    #
    #        wk := betak * sqrt(a/b) * (2*v[k]*v[k]' - J) * q 
    #
    #    with betak = W['beta'][k].
    # 
    # 3. The scaled variable:
    #
    #        lambda_k0 = sqrt(a*b) * c
    #        lambda_k1 = sqrt(a*b) * ( (2vk*vk' - J) * (-d*q + u/2) )_1
    #
    #    where 
    #
    #        u = st/a - J*zt/b 
    #        d = ( vk0 * (vk'*u) + u0/2 ) / (2*vk0 *(vk'*q) - q0 + 1).
    #
    # 4. Update scaling
    #   
    #        v[k] := wk^1/2 
    #              = 1 / sqrt(2*(wk0 + 1)) * (wk + e).
    #        beta[k] *=  sqrt(a/b)

    ind = m
    for k in range(len(W['v'])):

        v = W['v'][k]
        m = len(v)
        #print "-- start of q %d, ind = %d, m=%d v=\n" % (k, ind, m), strMat(v)

        # ln = sqrt( lambda_k' * J * lambda_k )
        ln = misc.jnrm2(lmbda, n = m, offset = ind) 

        # a = sqrt( sk' * J * sk ) = sqrt( st' * J * st ) 
        # s := s / a = st / a
        aa = misc.jnrm2(s, offset = ind, n = m)
        blas.scal(1.0/aa, s, offset = ind, n = m)
        #print "aa = %.17f, s=\n" % aa, strMat(s)

        # b = sqrt( zk' * J * zk ) = sqrt( zt' * J * zt )
        # z := z / a = zt / b
        bb = misc.jnrm2(z, offset = ind, n = m) 
        blas.scal(1.0/bb, z, offset = ind, n = m)
        #print "bb = %.17f, z=\n" % bb, strMat(s)

        # c = sqrt( ( 1 + (st'*zt) / (a*b) ) / 2 )
        cc = math.sqrt( ( 1.0 + blas.dot(s, z, offsetx = ind, offsety = 
            ind, n = m) ) / 2.0 )
        #print "cc = %.17f" % cc

        # vs = v' * st / a 
        vs = blas.dot(v, s, offsety = ind, n = m) 

        # vz = v' * J *zt / b
        vz = misc.jdot(v, z, offsety = ind, n = m) 
        #print "vs = %.17f, vz = %.17f" % (vs, vz)

        # vq = v' * q where q = (st/a + J * zt/b) / (2 * c)
        vq = (vs + vz ) / 2.0 / cc

        # vu = v' * u  where u =  st/a - J * zt/b 
        vu = vs - vz  

        # lambda_k0 = c
        lmbda[ind] = cc

        # wk0 = 2 * vk0 * (vk' * q) - q0 
        wk0 = 2 * v[0] * vq - ( s[ind] + z[ind] ) / 2.0 / cc 

        # d = (v[0] * (vk' * u) - u0/2) / (wk0 + 1)
        dd = (v[0] * vu - s[ind]/2.0 + z[ind]/2.0) / (wk0 + 1.0)
        #print "vq=%.17f vu=%.17f wk0=%.17f dd=%.17f" % (vq, vu, wk0, dd)

        # lambda_k1 = 2 * v_k1 * vk' * (-d*q + u/2) - d*q1 + u1/2
        blas.copy(v, lmbda, offsetx = 1, offsety = ind+1, n = m-1)
        blas.scal(2.0 * (-dd * vq + 0.5 * vu), lmbda, offset = ind+1, 
           n = m-1)
        blas.axpy(s, lmbda, 0.5 * (1.0 - dd/cc), offsetx = ind+1, offsety 
           = ind+1, n = m-1)
        blas.axpy(z, lmbda, 0.5 * (1.0 + dd/cc), offsetx = ind+1, offsety
           = ind+1, n = m-1)

        # Scale so that sqrt(lambda_k' * J * lambda_k) = sqrt(aa*bb).
        blas.scal(math.sqrt(aa*bb), lmbda, offset = ind, n = m)
            
        # v := (2*v*v' - J) * q 
        #    = 2 * (v'*q) * v' - (J* st/a + zt/b) / (2*c)
        blas.scal(2.0 * vq, v)
        #print "-- v[0] %.17f - %.17f" % (v[0], s[ind]/2.0/cc)
        v[0] -= s[ind] / 2.0 / cc
        #print "-- v[0] = %.17f", v[0]
        blas.axpy(s, v,  0.5/cc, offsetx = ind+1, offsety = 1, n = m-1)
        blas.axpy(z, v, -0.5/cc, offsetx = ind, n = m)

        # v := v^{1/2} = 1/sqrt(2 * (v0 + 1)) * (v + e)
        v[0] += 1.0
        blas.scal(1.0 / math.sqrt(2.0 * v[0]), v)
        #print "v=\n", strMat(v)

        # beta[k] *= ( aa / bb )**1/2
        W['beta'][k] *= math.sqrt( aa / bb )
            
        ind += m
        #print "-- end of q:\nz=\n", strMat(z), "\nlmbda=\n", strMat(lmbda)


    # 's' blocks
    # 
    # Let st, zt be the updated variables in the old scaling:
    # 
    #     st = Ls * Ls', zt = Lz * Lz'.
    #
    # where Ls and Lz are the 's' components of s, z.
    #
    # 1.  SVD Lz'*Ls = Uk * lambda_k^+ * Vk'.
    #
    # 2.  New scaling is 
    #
    #         r[k] := r[k] * Ls * Vk * diag(lambda_k^+)^{-1/2}
    #         rti[k] := r[k] * Lz * Uk * diag(lambda_k^+)^{-1/2}.
    #

    work = matrix(0.0, (max( [0] + [r.size[0] for r in W['r']])**2, 1))
    ind = mnl + ml + sum([ len(v) for v in W['v'] ])
    ind2, ind3 = ind, 0
    for k in range(len(W['r'])):
        r, rti = W['r'][k], W['rti'][k]
        m = r.size[0]
        
        # r := r*sk = r*Ls
        blas.gemm(r, s, work, m = m, n = m, k = m, ldB = m, ldC = m,
            offsetB = ind2)
        blas.copy(work, r, n = m**2)

        # rti := rti*zk = rti*Lz
        blas.gemm(rti, z, work, m = m, n = m, k = m, ldB = m, ldC = m,
            offsetB = ind2)
        blas.copy(work, rti, n = m**2)

        # SVD Lz'*Ls = U * lmbds^+ * V'; store U in sk and V' in zk.
        blas.gemm(z, s, work, transA = 'T', m = m, n = m, k = m, ldA = m,
            ldB = m, ldC = m, offsetA = ind2, offsetB = ind2)
        lapack.gesvd(work, lmbda, jobu = 'A', jobvt = 'A', m = m, n = m, 
            ldA = m, U = s, Vt = z, ldU = m, ldVt = m, offsetS = ind, 
            offsetU = ind2, offsetVt = ind2)

        # r := r*V
        blas.gemm(r, z, work, transB = 'T', m = m, n = m, k = m, ldB = m,
            ldC = m, offsetB = ind2)
        blas.copy(work, r, n = m**2)

        # rti := rti*U
        blas.gemm(rti, s, work, n = m, m = m, k = m, ldB = m, ldC = m,
            offsetB = ind2)
        blas.copy(work, rti, n = m**2)

        # r := r*lambda^{-1/2}; rti := rti*lambda^{-1/2}
        for i in range(m):    
            a = 1.0 / math.sqrt(lmbda[ind+i])
            blas.scal(a, r, offset = m*i, n = m)
            blas.scal(a, rti, offset = m*i, n = m)

        ind += m
        ind2 += m*m
        ind3 += m





def kkt_ldl(G, dims, A, mnl = 0):
    """
    Solution of KKT equations by a dense LDL factorization of the 
    3 x 3 system.
    
    Returns a function that (1) computes the LDL factorization of
    
        [ H           A'   GG'*W^{-1} ] 
        [ A           0    0          ],
        [ W^{-T}*GG   0   -I          ] 
    
    given H, Df, W, where GG = [Df; G], and (2) returns a function for 
    solving 
    
        [ H     A'   GG'   ]   [ ux ]   [ bx ]
        [ A     0    0     ] * [ uy ] = [ by ].
        [ GG    0   -W'*W  ]   [ uz ]   [ bz ]
    
    H is n x n,  A is p x n, Df is mnl x n, G is N x n where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ).
    """
    
    p, n = A.size
    ldK = n + p + mnl + dims['l'] + sum(dims['q']) + sum([ k*(k+1)/2 for k 
        in dims['s'] ])
    K = matrix(0.0, (ldK, ldK))
    ipiv = matrix(0, (ldK, 1))
    u = matrix(0.0, (ldK, 1))
    g = matrix(0.0, (mnl + G.size[0], 1))
    print "dims: ", str(dims)

    def factor(W, H = None, Df = None):

        blas.scal(0.0, K)
        if H is not None: K[:n, :n] = H
        K[n:n+p, :n] = A
        for k in xrange(n):
            if mnl: g[:mnl] = Df[:,k]
            g[mnl:] = G[:,k]
            scale(g, W, trans = 'T', inverse = 'I')
            misc.pack(g, K, dims, mnl, offsety = k*ldK + n + p)
        K[(ldK+1)*(p+n) :: ldK+1]  = -1.0
        lapack.sytrf(K, ipiv)

        def solve(x, y, z):

            # Solve
            #
            #     [ H          A'   GG'*W^{-1} ]   [ ux   ]   [ bx        ]
            #     [ A          0    0          ] * [ uy   [ = [ by        ]
            #     [ W^{-T}*GG  0   -I          ]   [ W*uz ]   [ W^{-T}*bz ]
            #
            # and return ux, uy, W*uz.
            #
            # On entry, x, y, z contain bx, by, bz.  On exit, they contain
            # the solution ux, uy, W*uz.
            blas.copy(x, u)
            blas.copy(y, u, offsety = n)
            scale(z, W, trans = 'T', inverse = 'I') 
            misc.pack(z, u, dims, mnl, offsety = n + p)
            lapack.sytrs(K, ipiv, u)
            blas.copy(u, x, n = n)
            blas.copy(u, y, offsetx = n, n = p)
            misc.unpack(u, z, dims, mnl, offsetx = n + p)
            #local_unpack(u, z, dims, mnl, offsetx = n + p)
            #print "** end solve **"
            #print "kkt-ldl solve end x=\n", str2(x, "%.17f")
            #print "kkt-ldl solve end z=\n", str2(z, "%.17f")
    
        return solve

    return factor

