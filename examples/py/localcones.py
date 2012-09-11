"""
Solver for linear and quadratic cone programs. 
"""

# Copyright 2010 L. Vandenberghe.
# Copyright 2004-2009 J. Dahl and L. Vandenberghe.
# 
# This file is part of CVXOPT version 1.1.3.
#
# CVXOPT is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# CVXOPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


__all__ = []
options = {}

import localmisc
import helpers

options = {}

def conelp(c, G, h, dims = None, A = None, b = None, primalstart = None, 
    dualstart = None, kktsolver = None, xnewcopy = None, xdot = None,
    xaxpy = None, xscal = None, ynewcopy = None, ydot = None, yaxpy = None,
    yscal = None):

    import math
    from cvxopt import base, blas, misc, matrix, spmatrix

    EXPON = 3
    STEP = 0.99

    try: DEBUG = options['debug']
    except KeyError: DEBUG = False

    try: MAXITERS = options['maxiters']
    except KeyError: MAXITERS = 100
    else:
        if type(MAXITERS) is not int or MAXITERS < 1:
           raise ValueError("options['maxiters'] must be a positive "\
               "integer")

    try: ABSTOL = options['abstol']
    except KeyError: ABSTOL = 1e-7
    else:
        if type(ABSTOL) is not float and type(ABSTOL) is not int:
            raise ValueError("options['abstol'] must be a scalar")

    try: RELTOL = options['reltol']
    except KeyError: RELTOL = 1e-6
    else:
        if type(RELTOL) is not float and type(RELTOL) is not int:
            raise ValueError("options['reltol'] must be a scalar")

    if RELTOL <= 0.0 and ABSTOL <= 0.0 :
        raise ValueError("at least one of options['reltol'] and " \
            "options['abstol'] must be positive")

    try: FEASTOL = options['feastol']
    except KeyError: FEASTOL = 1e-7
    else:
        if (type(FEASTOL) is not float and type(FEASTOL) is not int) or \
            FEASTOL <= 0.0:
            raise ValueError("options['feastol'] must be a positive "\
                "scalar")

    try: show_progress = options['show_progress']
    except KeyError: show_progress = True

    if kktsolver is None: 
        if dims and (dims['q'] or dims['s']):  
            kktsolver = 'qr'            
        else:
            kktsolver = 'chol2'
    defaultsolvers = ('ldl', 'ldl2', 'qr', 'chol', 'chol2')
    if type(kktsolver) is str and kktsolver not in defaultsolvers:
        raise ValueError("'%s' is not a valid value for kktsolver" \
            %kktsolver)

    # Argument error checking depends on level of customization.
    customkkt = type(kktsolver) is not str
    matrixG = type(G) in (matrix, spmatrix)
    matrixA = type(A) in (matrix, spmatrix)
    if (not matrixG or (not matrixA and A is not None)) and not customkkt:
        raise ValueError("use of function valued G, A requires a "\
            "user-provided kktsolver")
    customx = (xnewcopy != None or xdot != None or xaxpy != None or 
        xscal != None)
    if customx and (matrixG or matrixA or not customkkt):
        raise ValueError("use of non-vector type for x requires "\
            "function valued G, A and user-provided kktsolver")
    customy = (ynewcopy != None or ydot != None or yaxpy != None or 
        yscal != None)
    if customy and (matrixA or not customkkt):
        raise ValueError("use of non-vector type for y requires "\
            "function valued A and user-provided kktsolver")


    if not customx and (type(c) is not matrix or c.typecode != 'd' or 
        c.size[1] != 1):
        raise TypeError("'c' must be a 'd' matrix with one column")

    if type(h) is not matrix or h.typecode != 'd' or h.size[1] != 1:
        raise TypeError("'h' must be a 'd' matrix with 1 column")

    if not dims: dims = {'l': h.size[0], 'q': [], 's': []}
    if type(dims['l']) is not int or dims['l'] < 0: 
        raise TypeError("'dims['l']' must be a nonnegative integer")
    if [ k for k in dims['q'] if type(k) is not int or k < 1 ]:
        raise TypeError("'dims['q']' must be a list of positive integers")
    if [ k for k in dims['s'] if type(k) is not int or k < 0 ]:
        raise TypeError("'dims['s']' must be a list of nonnegative " \
            "integers")

    try: refinement = options['refinement']
    except KeyError: 
        if dims['q'] or dims['s']: refinement = 1
        else: refinement = 0
    else:
        if type(refinement) is not int or refinement < 0: 
            raise ValueError("options['refinement'] must be a "\
                "nonnegative integer")


    cdim = dims['l'] + sum(dims['q']) + sum([k**2 for k in dims['s']])
    cdim_pckd = dims['l'] + sum(dims['q']) + sum([k*(k+1)/2 for k in 
        dims['s']])
    cdim_diag = dims['l'] + sum(dims['q']) + sum(dims['s'])

    if h.size[0] != cdim:
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %cdim)

    # Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G.
    indq = [ dims['l'] ]  
    for k in dims['q']:  indq = indq + [ indq[-1] + k ] 

    # Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G.
    inds = [ indq[-1] ]
    for k in dims['s']:  inds = inds + [ inds[-1] + k**2 ] 

    #print "** dims =", dims
    #print "** indq =", indq
    #print "** inds =", inds
    if matrixG:
        if G.typecode != 'd' or G.size != (cdim, c.size[0]):
            raise TypeError("'G' must be a 'd' matrix of size (%d, %d)"\
                %(cdim, c.size[0]))
        def Gf(x, y, trans = 'N', alpha = 1.0, beta = 0.0): 
            misc.sgemv(G, x, y, dims, trans = trans, alpha = alpha, 
                beta = beta)
    else: 
        Gf = G

    if A is None: 
        if customx or customy: 
            def A(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else: 
            A = spmatrix([], [], [], (0, c.size[0]))
            matrixA = True
    if matrixA:
        if A.typecode != 'd' or A.size[1] != c.size[0]:
            raise TypeError("'A' must be a 'd' matrix with %d columns "\
                %c.size[0])
        def Af(x, y, trans = 'N', alpha = 1.0, beta = 0.0): 
            base.gemv(A, x, y, trans = trans, alpha = alpha, beta = beta)
    else: 
        Af = A

    if not customy:
        if b is None: b = matrix(0.0, (0,1))
        if type(b) is not matrix or b.typecode != 'd' or b.size[1] != 1:
            raise TypeError("'b' must be a 'd' matrix with one column")
        if matrixA and b.size[0] != A.size[0]:
            raise TypeError("'b' must have length %d" %A.size[0])
    else:
        if b is None: 
            raise ValueError("use of non vector type for y requires b")


    # kktsolver(W) returns a routine for solving 3x3 block KKT system 
    #
    #     [ 0   A'  G'*W^{-1} ] [ ux ]   [ bx ]
    #     [ A   0   0         ] [ uy ] = [ by ].
    #     [ G   0   -W'       ] [ uz ]   [ bz ]

    if kktsolver in defaultsolvers:
        if b.size[0] > c.size[0] or b.size[0] + cdim_pckd < c.size[0]:
           raise ValueError("Rank(A) < p or Rank([G; A]) < n")
        if kktsolver == 'ldl': 
            factor = localmisc.kkt_ldl(G, dims, A)
        elif kktsolver == 'ldl2':
            factor = misc.kkt_ldl2(G, dims, A)
        elif kktsolver == 'qr':
            factor = misc.kkt_qr(G, dims, A)
        elif kktsolver == 'chol':
            factor = misc.kkt_chol(G, dims, A)
        else:
            factor = misc.kkt_chol2(G, dims, A)
        def kktsolver(W):
            return factor(W)


    # res() evaluates residual in 5x5 block KKT system
    #
    #     [ vx   ]    [ 0         ]   [ 0   A'  G'  c ] [ ux        ]
    #     [ vy   ]    [ 0         ]   [-A   0   0   b ] [ uy        ]
    #     [ vz   ] += [ W'*us     ] - [-G   0   0   h ] [ W^{-1}*uz ]
    #     [ vtau ]    [ dg*ukappa ]   [-c' -b' -h'  0 ] [ utau/dg   ]
    # 
    #           vs += lmbda o (dz + ds) 
    #       vkappa += lmbdg * (dtau + dkappa).

    ws3, wz3 = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1))
    helpers.sp_add_var("ws3", ws3)
    helpers.sp_add_var("wz3", wz3)
    def res(ux, uy, uz, utau, us, ukappa, vx, vy, vz, vtau, vs, vkappa, W,
        dg, lmbda):

        # vx := vx - A'*uy - G'*W^{-1}*uz - c*utau/dg
        Af(uy, vx, alpha = -1.0, beta = 1.0, trans = 'T')
        blas.copy(uz, wz3)
        localmisc.scale(wz3, W, inverse = 'I')
        Gf(wz3, vx, alpha = -1.0, beta = 1.0, trans = 'T')
        xaxpy(c, vx, alpha = -utau[0]/dg)

        # vy := vy + A*ux - b*utau/dg
        Af(ux, vy, alpha = 1.0, beta = 1.0)
        yaxpy(b, vy, alpha = -utau[0]/dg)
 
        # vz := vz + G*ux - h*utau/dg + W'*us
        Gf(ux, vz, alpha = 1.0, beta = 1.0)
        blas.axpy(h, vz, alpha = -utau[0]/dg)
        blas.copy(us, ws3)
        localmisc.scale(ws3, W, trans = 'T')
        blas.axpy(ws3, vz)

        # vtau := vtau + c'*ux + b'*uy + h'*W^{-1}*uz + dg*ukappa
        vtauplus = dg*ukappa[0] + xdot(c,ux) + ydot(b,uy) + \
            misc.sdot(h, wz3, dims) 
        vtau[0] += vtauplus

        # vs := vs + lmbda o (uz + us)
        blas.copy(us, ws3)
        blas.axpy(uz, ws3)
        #localmisc.local_sprod(ws3, lmbda, dims, diag = 'D')
        misc.sprod(ws3, lmbda, dims, diag = 'D')
        blas.axpy(ws3, vs)

        # vkappa += vkappa + lmbdag * (utau + ukappa)
        vkappa[0] += lmbda[-1] * (utau[0] + ukappa[0])


    if xnewcopy is None: xnewcopy = matrix 
    if xdot is None: xdot = blas.dot
    if xaxpy is None: xaxpy = blas.axpy 
    if xscal is None: xscal = blas.scal 
    def xcopy(x, y): 
        xscal(0.0, y) 
        xaxpy(x, y)
    if ynewcopy is None: ynewcopy = matrix 
    if ydot is None: ydot = blas.dot 
    if yaxpy is None: yaxpy = blas.axpy 
    if yscal is None: yscal = blas.scal
    def ycopy(x, y): 
        yscal(0.0, y) 
        yaxpy(x, y)

    resx0 = max(1.0, math.sqrt(xdot(c,c)))
    resy0 = max(1.0, math.sqrt(ydot(b,b)))
    resz0 = max(1.0, misc.snrm2(h, dims))

    # Select initial points.

    x = xnewcopy(c);  xscal(0.0, x)
    y = ynewcopy(b);  yscal(0.0, y)
    s, z = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1))
    dx, dy = xnewcopy(c), ynewcopy(b)
    ds, dz = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1))
    dkappa, dtau = matrix(0.0, (1,1)), matrix(0.0, (1,1))

    helpers.sp_add_var("x", x)
    helpers.sp_add_var("s", s)
    helpers.sp_add_var("z", z)
    helpers.sp_add_var("dx", dx)
    helpers.sp_add_var("ds", ds)
    helpers.sp_add_var("dz", dz)

    if primalstart is None or dualstart is None:

        # Factor
        #
        #     [ 0   A'  G' ] 
        #     [ A   0   0  ].
        #     [ G   0  -I  ]
    
        W = {}
        W['d'] = matrix(1.0, (dims['l'], 1)) 
        W['di'] = matrix(1.0, (dims['l'], 1)) 
        W['v'] = [ matrix(0.0, (m,1)) for m in dims['q'] ]
        W['beta'] = len(dims['q']) * [ 1.0 ] 
        for v in W['v']: v[0] = 1.0
        W['r'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
        W['rti'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
        for r in W['r']: r[::r.size[0]+1 ] = 1.0
        for rti in W['rti']: rti[::rti.size[0]+1 ] = 1.0
        try: f = kktsolver(W)
        except ArithmeticError:  
            raise ValueError("Rank(A) < p or Rank([G; A]) < n")

    if primalstart is None:

        # minimize    || G * x - h ||^2
        # subject to  A * x = b
        #
        # by solving
        #
        #     [ 0   A'  G' ]   [ x  ]   [ 0 ]
        #     [ A   0   0  ] * [ dy ] = [ b ].
        #     [ G   0  -I  ]   [ -s ]   [ h ]

        xscal(0.0, x)
        ycopy(b, dy)  
        blas.copy(h, s)
        try: f(x, dy, s) 
        except ArithmeticError:  
            raise ValueError("Rank(A) < p or Rank([G; A]) < n")
        blas.scal(-1.0, s)  
        #print "** initial s=\n", s
    else:
        xcopy(primalstart['x'], x)
        blas.copy(primalstart['s'], s)

    # ts = min{ t | s + t*e >= 0 }
    ts = misc.max_step(s, dims)
    #print "** initial ts: ", ts
    if ts >= 0 and primalstart: 
        raise ValueError("initial s is not positive")


    if dualstart is None:

        # minimize   || z ||^2
        # subject to G'*z + A'*y + c = 0
        #
        # by solving
        #
        #     [ 0   A'  G' ] [ dx ]   [ -c ]
        #     [ A   0   0  ] [ y  ] = [  0 ].
        #     [ G   0  -I  ] [ z  ]   [  0 ]

        xcopy(c, dx); 
        xscal(-1.0, dx)
        yscal(0.0, y)
        blas.scal(0.0, z)
        try: f(dx, y, z)
        except ArithmeticError:  
            raise ValueError("Rank(A) < p or Rank([G; A]) < n")

        #print "initial z=\n", localmisc.strMat(z)
        #print "** initial z=\n", z 
    else:
        if 'y' in dualstart: ycopy(dualstart['y'], y)
        blas.copy(dualstart['z'], z)

    # tz = min{ t | z + t*e >= 0 }
    tz = misc.max_step(z, dims)
    #print "** initial tz: ", tz
    if tz >= 0 and dualstart: 
        raise ValueError("initial z is not positive")

    nrms = misc.snrm2(s, dims)
    nrmz = misc.snrm2(z, dims)
    #print "** nrms=%.17f nrmz=%.17f" %(nrms, nrmz)
    #print "** ts  =%.17f tz  =%.17f" %(ts, tz)

    if primalstart is None and dualstart is None: 

        gap = misc.sdot(s, z, dims) 
        pcost = xdot(c,x)
        dcost = -ydot(b,y) - misc.sdot(h, z, dims) 
        if pcost < 0.0:
            relgap = gap / -pcost
        elif dcost > 0.0:
            relgap = gap / dcost
        else: 
            relgap = None

        if ts <= 0 and tz <= 0 and (gap <= ABSTOL or ( relgap is not None
            and relgap <= RELTOL )):

            # The initial points we constructed happen to be feasible and 
            # optimal.  

            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(s, m, ind)
                misc.symm(z, m, ind)
                ind += m**2

            # rx = A'*y + G'*z + c
            rx = xnewcopy(c)
            Af(y, rx, beta = 1.0, trans = 'T') 
            Gf(z, rx, beta = 1.0, trans = 'T') 
            resx = math.sqrt( xdot(rx, rx) ) 

            # ry = b - A*x 
            ry = ynewcopy(b)
            Af(x, ry, alpha = -1.0, beta = 1.0)
            resy = math.sqrt( ydot(ry, ry) ) 

            # rz = s + G*x - h 
            rz = matrix(0.0, (cdim,1))
            Gf(x, rz)
            blas.axpy(s, rz)
            blas.axpy(h, rz, alpha = -1.0)
            resz = misc.snrm2(rz, dims) 

            pres = max(resy/resy0, resz/resz0)
            dres = resx/resx0
            cx, by, hz = xdot(c,x), ydot(b,y), misc.sdot(h, z, dims) 

            if show_progress:
                print("Optimal solution found.")
            return { 'x': x, 'y': y, 's': s, 'z': z,
                'status': 'optimal', 
                'gap': gap, 
                'relative gap': relgap, 
                'primal objective': cx,
                'dual objective': -(by + hz),
                'primal infeasibility': pres,
                'primal slack': -ts,
                'dual slack': -tz,
                'dual infeasibility': dres,
                'residual as primal infeasibility certificate': None,
                'residual as dual infeasibility certificate': None,
                'iterations': 0 } 

        if ts >= -1e-8 * max(nrms, 1.0):  
            a = 1.0 + ts  
            s[:dims['l']] += a
            s[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                s[ind : ind+m*m : m+1] += a
                ind += m**2
            #print "indq: ", indq
            #print "scaled s=\n", localmisc.strMat(s)

        if tz >= -1e-8 * max(nrmz, 1.0):
            a = 1.0 + tz  
            z[:dims['l']] += a
            z[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                z[ind : ind+m*m : m+1] += a
                ind += m**2
            #print "scaled z=\n", localmisc.strMat(z)


    elif primalstart is None and dualstart is not None:

        if ts >= -1e-8 * max(nrms, 1.0):  
            a = 1.0 + ts  
            s[:dims['l']] += a
            s[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                s[ind : ind+m*m : m+1] += a
                ind += m**2

    elif primalstart is not None and dualstart is None:

        if tz >= -1e-8 * max(nrmz, 1.0):
            a = 1.0 + tz  
            z[:dims['l']] += a
            z[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                z[ind : ind+m*m : m+1] += a
                ind += m**2


    tau, kappa = 1.0, 1.0

    rx, hrx = xnewcopy(c), xnewcopy(c)
    ry, hry = ynewcopy(b), ynewcopy(b)
    rz, hrz = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1))
    sigs = matrix(0.0, (sum(dims['s']), 1))
    sigz = matrix(0.0, (sum(dims['s']), 1))
    lmbda = matrix(0.0, (cdim_diag + 1, 1))
    lmbdasq = matrix(0.0, (cdim_diag + 1, 1)) 

    #print "pre-gap s=\n", s
    #print "pre-gap z=\n", z
    gap = misc.sdot(s, z, dims) 

    #print "** iterate %d times [gap=%.4f] ..." % (MAXITERS+1, gap)
    #print "preloop x=\n", localmisc.str2(x, "%.17f")
    #print "preloop s=\n", localmisc.str2(s, "%.17f")
    #print "preloop z=\n", localmisc.str2(z, "%.17f")
    helpers.sp_add_var("lmbda", lmbda)
    helpers.sp_add_var("lmbdasq", lmbdasq)
    helpers.sp_add_var("rx", rx)
    helpers.sp_add_var("rz", rz)

    for iters in xrange(MAXITERS+1):
        helpers.sp_major_next()
        helpers.sp_create("loop-start", 100)

        # hrx = -A'*y - G'*z 
        Af(y, hrx, alpha = -1.0, trans = 'T') 
        #print "Af hrx=\n", localmisc.strMat(hrx)
        Gf(z, hrx, alpha = -1.0, beta = 1.0, trans = 'T') 
        hresx = math.sqrt( xdot(hrx, hrx) ) 
        #print "Gf hrx=\n", localmisc.strMat(hrx)
        #print "hresx =", hresx

        # rx = hrx - c*tau 
        #    = -A'*y - G'*z - c*tau
        xcopy(hrx, rx)
        xaxpy(c, rx, alpha = -tau)
        resx = math.sqrt( xdot(rx, rx) ) / tau
        #print "initial rx=\n", localmisc.strMat(rx)
        #print "resx =", resx

        # hry = A*x  
        Af(x, hry)
        hresy = math.sqrt( ydot(hry, hry) )
        #print "hresy =", hresy

        # ry = hry - b*tau 
        #    = A*x - b*tau
        ycopy(hry, ry)
        yaxpy(b, ry, alpha = -tau)
        resy = math.sqrt( ydot(ry, ry) ) / tau
        #print "resy =", resy

        # hrz = s + G*x  
        Gf(x, hrz)
        blas.axpy(s, hrz)
        hresz = misc.snrm2(hrz, dims) 
        #print "hresz =", hresz

        # rz = hrz - h*tau 
        #    = s + G*x - h*tau
        blas.scal(0, rz)
        blas.axpy(hrz, rz)
        blas.axpy(h, rz, alpha = -tau)
        resz = misc.snrm2(rz, dims) / tau 
        #print "resz =", resz

        # rt = kappa + c'*x + b'*y + h'*z 
        cx, by, hz = xdot(c,x), ydot(b,y), misc.sdot(h, z, dims) 
        rt = kappa + cx + by + hz 

        # Statistics for stopping criteria.
        pcost, dcost = cx / tau, -(by + hz) / tau        
        if pcost < 0.0:
            relgap = gap / -pcost
        elif dcost > 0.0:
            relgap = gap / dcost
        else: 
            relgap = None
        pres = max(resy/resy0, resz/resz0)
        dres = resx/resx0
        if hz + by < 0.0:  
           pinfres =  hresx / resx0 / (-hz - by) 
        else:
           pinfres =  None
        if cx < 0.0: 
           dinfres = max(hresy / resy0, hresz/resz0) / (-cx) 
        else:
           dinfres = None

        if show_progress:
            if iters == 0:
                print("% 10s% 12s% 10s% 8s% 7s % 5s" %("pcost", "dcost",
                    "gap", "pres", "dres", "k/t"))
            print("%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e% 7.0e" \
                %(iters, pcost, dcost, gap, pres, dres, kappa/tau))

        helpers.sp_create("isready", 200, {"hresx": hresx,
                                           "resx": resx,
                                           "hresy": hresy,
                                           "resy": resy,
                                           "resz": resz,
                                           "hresz": hresz,
                                           "cx": cx,
                                           "by": by,
                                           "rt": rt,
                                           "hz": hz,
                                           "pres": pres,
                                           "dres": dres,
                                           "gap": gap
                                           })

        if ( pres <= FEASTOL and dres <= FEASTOL and ( gap <= ABSTOL or 
            (relgap is not None and relgap <= RELTOL) ) ) or \
            iters == MAXITERS:
            xscal(1.0/tau, x)
            yscal(1.0/tau, y)
            blas.scal(1.0/tau, s)
            blas.scal(1.0/tau, z)
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(s, m, ind)
                misc.symm(z, m, ind)
                ind += m**2
            ts = misc.max_step(s, dims)
            tz = misc.max_step(z, dims)
            if iters == MAXITERS:
                if show_progress:
                    print("Terminated (maximum number of iterations "\
                        "reached).")
                return { 'x': x, 'y': y, 's': s, 'z': z,
                    'status': 'unknown', 
                    'gap': gap, 
                    'relative gap': relgap, 
                    'primal objective': pcost,
                    'dual objective' : dcost,
                    'primal infeasibility': pres,
                    'dual infeasibility': dres,
                    'primal slack': -ts,
                    'dual slack': -tz,
                    'residual as primal infeasibility certificate': 
                        pinfres,
                    'residual as dual infeasibility certificate': 
                        dinfres,
                    'iterations': iters}

            else:
                if show_progress:
                    print("Optimal solution found.")
                return { 'x': x, 'y': y, 's': s, 'z': z,
                    'status': 'optimal', 
                    'gap': gap, 
                    'relative gap': relgap, 
                    'primal objective': pcost,
                    'dual objective' : dcost,
                    'primal infeasibility': pres,
                    'dual infeasibility': dres,
                    'primal slack': -ts,
                    'dual slack': -tz,
                    'residual as primal infeasibility certificate': None,
                    'residual as dual infeasibility certificate': None,
                    'iterations': iters }

        elif pinfres is not None and pinfres <= FEASTOL:
            yscal(1.0/(-hz - by), y)
            blas.scal(1.0/(-hz - by), z)
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(z, m, ind)
                ind += m**2
            tz = misc.max_step(z, dims)
            if show_progress:
                print("Certificate of primal infeasibility found.")
            return { 'x': None, 'y': y, 's': None, 'z': z,
                'status': 'primal infeasible',
                'gap': None, 
                'relative gap': None, 
                'primal objective': None,
                'dual objective' : 1.0,
                'primal infeasibility': None,
                'dual infeasibility': None,
                'primal slack': None,
                'dual slack': -tz,
                'residual as primal infeasibility certificate': pinfres,
                'residual as dual infeasibility certificate': None,
                'iterations': iters }

        elif dinfres is not None and dinfres <= FEASTOL:
            xscal(1.0/(-cx), x)
            blas.scal(1.0/(-cx), s)
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(s, m, ind)
                ind += m**2
            y, z = None, None
            ts = misc.max_step(s, dims)
            if show_progress:
                print("Certificate of dual infeasibility found.")
            return {'x': x, 'y': None, 's': s, 'z': None,
                'status': 'dual infeasible',
                'gap': None, 
                'relative gap': None, 
                'primal objective': -1.0,
                'dual objective' : None,
                'primal infeasibility': None,
                'dual infeasibility': None,
                'primal slack': -ts,
                'dual slack': None,
                'residual as primal infeasibility certificate': None,
                'residual as dual infeasibility certificate': dinfres,
                'iterations': iters }


        # Compute initial scaling W:
        # 
        #     W * z = W^{-T} * s = lambda
        #     dg * tau = 1/dg * kappa = lambdag.

        if iters == 0:
            
            #print "compute scaling: lmbda=\n",localmisc.strMat(lmbda)
            #print "s=\n", localmisc.strMat(s)
            #print "z=\n", localmisc.strMat(z)
            W = localmisc.local_compute_scaling(s, z, lmbda, dims, mnl = 0)
            helpers.sp_add_var("W", W)
            #     dg = sqrt( kappa / tau )
            #     dgi = sqrt( tau / kappa )
            #     lambda_g = sqrt( tau * kappa )  
            # 
            # lambda_g is stored in the last position of lmbda.
    
            dg = math.sqrt( kappa / tau )
            dgi = math.sqrt( tau / kappa )
            lmbda[-1] = math.sqrt( tau * kappa )
            #print "lmbda=\n", localmisc.strMat(lmbda)
            #localmisc.printW(W)
            helpers.sp_create("compute_scaling", 300)

        # lmbdasq := lmbda o lmbda 
        misc.ssqr(lmbdasq, lmbda, dims)
        lmbdasq[-1] = lmbda[-1]**2
        #print "lmbdasq=\n",localmisc.strMat(lmbdasq)


        # f3(x, y, z) solves    
        #
        #     [ 0  A'  G'   ] [ ux        ]   [ bx ]
        #     [ A  0   0    ] [ uy        ] = [ by ].
        #     [ G  0  -W'*W ] [ W^{-1}*uz ]   [ bz ]
        #
        # On entry, x, y, z contain bx, by, bz.
        # On exit, they contain ux, uy, uz.
        #
        # Also solve
        #
        #     [ 0   A'  G'    ] [ x1        ]          [ c ]
        #     [-A   0   0     ]*[ y1        ] = -dgi * [ b ].
        #     [-G   0   W'*W  ] [ W^{-1}*z1 ]          [ h ]
         

        try: 
            f3 = kktsolver(W)
            if iters == 0:
                x1, y1 = xnewcopy(c), ynewcopy(b)
                z1 = matrix(0.0, (cdim,1))
            xcopy(c, x1);  xscal(-1, x1)
            ycopy(b, y1)
            blas.copy(h, z1)
            f3(x1, y1, z1)
            #print "f3-result: x1=\n", x1
            #print "f3-result: z1=\n", z1
            xscal(dgi, x1)
            yscal(dgi, y1)
            blas.scal(dgi, z1)
        except ArithmeticError:
            if iters == 0 and primalstart and dualstart: 
                raise ValueError("Rank(A) < p or Rank([G; A]) < n")
            else:
                xscal(1.0/tau, x)
                yscal(1.0/tau, y)
                blas.scal(1.0/tau, s)
                blas.scal(1.0/tau, z)
                ind = dims['l'] + sum(dims['q'])
                for m in dims['s']:
                    misc.symm(s, m, ind)
                    misc.symm(z, m, ind)
                    ind += m**2
                ts = misc.max_step(s, dims)
                tz = misc.max_step(z, dims)
                if show_progress:
                    print("Terminated (singular KKT matrix).")
                return { 'x': x, 'y': y, 's': s, 'z': z,
                    'status': 'unknown', 
                    'gap': gap, 
                    'relative gap': relgap, 
                    'primal objective': pcost,
                    'dual objective' : dcost,
                    'primal infeasibility': pres,
                    'dual infeasibility': dres,
                    'primal slack': -ts,
                    'dual slack': -tz,
                    'residual as primal infeasibility certificate': 
                        pinfres,
                    'residual as dual infeasibility certificate': 
                        dinfres,
                    'iterations': iters }


        # f6_no_ir(x, y, z, tau, s, kappa) solves
        #
        #     [ 0         ]   [  0   A'  G'  c ] [ ux        ]    [ bx   ]
        #     [ 0         ]   [ -A   0   0   b ] [ uy        ]    [ by   ]
        #     [ W'*us     ] - [ -G   0   0   h ] [ W^{-1}*uz ] = -[ bz   ]
        #     [ dg*ukappa ]   [ -c' -b' -h'  0 ] [ utau/dg   ]    [ btau ]
        # 
        #     lmbda o (uz + us) = -bs
        #     lmbdag * (utau + ukappa) = -bkappa.
        #
        # On entry, x, y, z, tau, s, kappa contain bx, by, bz, btau, 
        # bkappa.  On exit, they contain ux, uy, uz, utau, ukappa.

        # th = W^{-T} * h
        if iters == 0:
            th = matrix(0.0, (cdim,1))
            helpers.sp_add_var("th", th)
        blas.copy(h, th)
        localmisc.scale(th, W, trans = 'T', inverse = 'I')
        #print "th=\n", th

        def f6_no_ir(x, y, z, tau, s, kappa):

            # Solve 
            #
            #     [  0   A'  G'    0   ] [ ux        ]   
            #     [ -A   0   0     b   ] [ uy        ]  
            #     [ -G   0   W'*W  h   ] [ W^{-1}*uz ] 
            #     [ -c' -b' -h'    k/t ] [ utau/dg   ]
            #
            #           [ bx                    ]
            #           [ by                    ]
            #         = [ bz - W'*(lmbda o\ bs) ]
            #           [ btau - bkappa/tau     ]
            #
            #     us = -lmbda o\ bs - uz
            #     ukappa = -bkappa/lmbdag - utau.


            # First solve 
            #
            #     [ 0  A' G'   ] [ ux        ]   [  bx                    ]
            #     [ A  0  0    ] [ uy        ] = [ -by                    ]
            #     [ G  0 -W'*W ] [ W^{-1}*uz ]   [ -bz + W'*(lmbda o\ bs) ]

            minor = helpers.sp_minor_top()
            # y := -y = -by
            yscal(-1.0, y) 

            # s := -lmbda o\ s = -lmbda o\ bs
            misc.sinv(s, lmbda, dims)
            blas.scal(-1.0, s)

            # z := -(z + W'*s) = -bz + W'*(lambda o\ bs)
            blas.copy(s, ws3)  
            helpers.sp_create("prescale", minor+5)
            helpers.sp_minor_push(minor+5)
            #misc.scale(ws3, W, trans = 'T')
            localmisc.scale(ws3, W, trans = 'T')
            helpers.sp_minor_pop()
            blas.axpy(ws3, z)
            blas.scal(-1.0, z)

            helpers.sp_create("f3-call", minor+20)
            # Solve system.
            helpers.sp_minor_push(minor+20)
            f3(x, y, z)
            helpers.sp_minor_pop()
            helpers.sp_create("f3-return", minor+40)

            # Combine with solution of 
            #
            #     [ 0   A'  G'    ] [ x1         ]          [ c ]
            #     [-A   0   0     ] [ y1         ] = -dgi * [ b ]
            #     [-G   0   W'*W  ] [ W^{-1}*dzl ]          [ h ]
            # 
            # to satisfy
            #
            #     -c'*x - b'*y - h'*W^{-1}*z + dg*tau = btau - bkappa/tau.

            # kappa[0] := -kappa[0] / lmbd[-1] = -bkappa / lmbdag
            kappa[0] = -kappa[0] / lmbda[-1]
            # tau[0] = tau[0] + kappa[0] / dgi = btau[0] - bkappa / tau
            tau[0] += kappa[0] / dgi
 
            tau[0] = dgi * ( tau[0] + xdot(c,x) + ydot(b,y) + 
                misc.sdot(th, z, dims) ) / (1.0 + misc.sdot(z1, z1, dims))
            xaxpy(x1, x, alpha = tau[0])
            yaxpy(y1, y, alpha = tau[0])
            blas.axpy(z1, z, alpha = tau[0])

            # s := s - z = - lambda o\ bs - z 
            blas.axpy(z, s, alpha = -1)

            kappa[0] -= tau[0]


        # f6(x, y, z, tau, s, kappa) solves the same system as f6_no_ir, 
        # but applies iterative refinement.

        if iters == 0:
            if refinement or DEBUG:
                wx, wy = xnewcopy(c), ynewcopy(b)
                wz, ws = matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1))
                wtau, wkappa = matrix(0.0), matrix(0.0)
                helpers.sp_add_var("wx", wx)
                helpers.sp_add_var("ws", ws)
                helpers.sp_add_var("wz", wz)
            if refinement:
                wx2, wy2 = xnewcopy(c), ynewcopy(b)
                wz2, ws2 = matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1))
                wtau2, wkappa2 = matrix(0.0), matrix(0.0)
                helpers.sp_add_var("wx2", wx2)
                helpers.sp_add_var("ws2", ws2)
                helpers.sp_add_var("wz2", wz2)

        def f6(x, y, z, tau, s, kappa):
            minor = helpers.sp_minor_top()
            helpers.sp_create("startf6", minor+100)
            if refinement or DEBUG:
                xcopy(x, wx)
                ycopy(y, wy)
                blas.copy(z, wz)
                wtau[0] = tau[0]
                blas.copy(s, ws)
                wkappa[0] = kappa[0]
            helpers.sp_create("pref6_no_ir", minor+200)
            f6_no_ir(x, y, z, tau, s, kappa)
            helpers.sp_create("postf6_no_ir", minor+399)

            for i in xrange(refinement):
                xcopy(wx, wx2)
                ycopy(wy, wy2)
                blas.copy(wz, wz2)
                wtau2[0] = wtau[0]
                blas.copy(ws, ws2)
                wkappa2[0] = wkappa[0]
                helpers.sp_create("res-call", minor+400)
                helpers.sp_minor_push(minor+400)
                res(x, y, z, tau, s, kappa, wx2, wy2, wz2, wtau2, ws2, 
                    wkappa2, W, dg, lmbda)
                helpers.sp_minor_pop()

                helpers.sp_create("refine_pref6_no_ir", minor+500)
                helpers.sp_minor_push(minor+500)
                f6_no_ir(wx2, wy2, wz2, wtau2, ws2, wkappa2)
                helpers.sp_minor_pop()
                helpers.sp_create("refine_postf6_no_ir", minor+600)

                xaxpy(wx2, x)
                yaxpy(wy2, y)
                blas.axpy(wz2, z)
                tau[0] += wtau2[0]
                blas.axpy(ws2, s)
                kappa[0] += wkappa2[0]
                #print "refinement: tau=%.17f" % tau[0], " kappa=%.17f" % kappa[0]

            #print "== end of f6 .."
            if DEBUG:
                helpers.sp_minor_push(minor+700)
                res(x, y, z, tau, s, kappa, wx, wy, wz, wtau, ws, wkappa,
                    W, dg, lmbda)
                helpers.sp_minor_pop()
                print("KKT residuals")
                print("    'x': %.6e" %math.sqrt(xdot(wx, wx)))
                print("    'y': %.6e" %math.sqrt(ydot(wy, wy)))
                print("    'z': %.6e" %misc.snrm2(wz, dims))
                print("    'tau': %.6e" %abs(wtau[0]))
                print("    's': %.6e" %misc.snrm2(ws, dims))
                print("    'kappa': %.6e" %abs(wkappa[0]))
 

        mu = blas.nrm2(lmbda)**2 / (1 + cdim_diag) 
        sigma = 0.0
        #print "** mu = %.4f" % mu
        for i in [0,1]:
            #print "--- loop [0,1] start ---"

            # Solve
            #
            #     [ 0         ]   [  0   A'  G'  c ] [ dx        ]
            #     [ 0         ]   [ -A   0   0   b ] [ dy        ]
            #     [ W'*ds     ] - [ -G   0   0   h ] [ W^{-1}*dz ]
            #     [ dg*dkappa ]   [ -c' -b' -h'  0 ] [ dtau/dg   ]
            #
            #                       [ rx   ]
            #                       [ ry   ]
            #         = - (1-sigma) [ rz   ]
            #                       [ rtau ]
            #
            #     lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e
            #     lmbdag * (dtau + dkappa) = - kappa * tau + sigma*mu

            
            # ds = -lmbdasq if i is 0
            #    = -lmbdasq - dsa o dza + sigma*mu*e if i is 1
            # dkappa = -lambdasq[-1] if i is 0 
            #        = -lambdasq[-1] - dkappaa*dtaua + sigma*mu if i is 1.

            blas.copy(lmbdasq, ds, n = dims['l'] + sum(dims['q']))
            ind = dims['l'] + sum(dims['q'])
            ind2 = ind
            blas.scal(0.0, ds, offset = ind)
            #print "** i=%d, ds =\n" % i, ds
            for m in dims['s']:
                blas.copy(lmbdasq, ds, n = m, offsetx = ind2, 
                    offsety = ind, incy = m+1)
                ind += m*m
                ind2 += m
            dkappa[0] = lmbdasq[-1]
            #print "dkappa[0] = %.17f" % dkappa[0]

            if i == 1:
                #print "scaling with sigma*mu (%.17f,%.17f)" % (sigma, mu)
                blas.axpy(ws3, ds)
                ds[:dims['l']] -= sigma*mu 
                #print "** sigmaMu scaling indexes", indq[:-1]
                ds[indq[:-1]] -= sigma*mu
                ind = dims['l'] + sum(dims['q'])
                ind2 = ind
                for m in dims['s']:
                    ds[ind : ind+m*m : m+1] -= sigma*mu
                    ind += m*m
                dkappa[0] += wkappa3 - sigma*mu
                #print "dtau=%.17f" % dtau[0], " dkappa=%.17f" % dkappa[0]
 
            # (dx, dy, dz, dtau) = (1-sigma)*(rx, ry, rz, rt)
            xcopy(rx, dx);  xscal(1.0 - sigma, dx)
            ycopy(ry, dy);  yscal(1.0 - sigma, dy)
            blas.copy(rz, dz);  blas.scal(1.0 - sigma, dz)
            dtau[0] = (1.0 - sigma) * rt 
            #print "setting: dtau=%.17f, sigma=%.17f, rt=%.17f" % (dtau[0], sigma, rt)

            helpers.sp_create("pref6", (1+i)*1000)
            helpers.sp_minor_push((1+i)*1000)
            f6(dx, dy, dz, dtau, ds, dkappa)
            helpers.sp_minor_pop()
            helpers.sp_create("postf6", (1+i)*1000+800)
            
            # Save ds o dz and dkappa * dtau for Mehrotra correction
            if i == 0:
                blas.copy(ds, ws3)
                misc.sprod(ws3, dz, dims)
                wkappa3 = dtau[0] * dkappa[0]

            # Maximum step to boundary.
            #
            # If i is 1, also compute eigenvalue decomposition of the 's' 
            # blocks in ds, dz.  The eigenvectors Qs, Qz are stored in 
            # dsk, dzk.  The eigenvalues are stored in sigs, sigz. 

            helpers.sp_minor_push((1+i)*1000+900)
            localmisc.scale2(lmbda, ds, dims)
            localmisc.scale2(lmbda, dz, dims)
            helpers.sp_minor_pop()
            helpers.sp_create("post-scale2", (1+i)*1000+990)
            if i == 0:
                ts = misc.max_step(ds, dims)
                tz = misc.max_step(dz, dims)
            else:
                ts = misc.max_step(ds, dims, sigma = sigs)
                tz = misc.max_step(dz, dims, sigma = sigz)

            tt = -dtau[0] / lmbda[-1]
            tk = -dkappa[0] / lmbda[-1]
            t = max([ 0.0, ts, tz, tt, tk ])
            #print "-- max t=%.17f from : " % t, str([0.0, ts, tz, tt, tk])
            if t == 0.0:
                step = 1.0
            else:
                if i == 0:
                    step = min(1.0, 1.0 / t)
                else:
                    step = min(1.0, STEP / t)
            if i == 0:
                sigma = (1.0 - step)**EXPON
            #print "--- loop [0,1] end ---"

        #print "** tau = %.17f, kappa = %.17f" % (tau, kappa)
        #print "** step = %.17f, sigma = %.17f" % (step, sigma)
        #print "-- post loop lmbda=\n", lmbda, "ds=\n", ds, "dz=\n", dz
        #print "-- post loop end ---"
        #print "dx=\n", localmisc.strMat(dx), "\ndy=\n", localmisc.strMat(dy)
        #print "ds=\n", localmisc.strMat(ds), "\ndz=\n", localmisc.strMat(dz)
        #print "sigs=\n", localmisc.strMat(sigs), "\nsigz=\n", localmisc.strMat(sigz)

        helpers.sp_create("update-xy", 7000)
        # Update x, y.
        xaxpy(dx, x, alpha = step)
        yaxpy(dy, y, alpha = step)
        #print "update x=\n", localmisc.strMat(x), "\ny=\n", localmisc.strMat(y)

        # Replace 'l' and 'q' blocks of ds and dz with the updated 
        # variables in the current scaling.
        # Replace 's' blocks of ds and dz with the factors Ls, Lz in a 
        # factorization Ls*Ls', Lz*Lz' of the updated variables in the 
        # current scaling.

        # ds := e + step*ds for 'l' and 'q' blocks.
        # dz := e + step*dz for 'l' and 'q' blocks.
        blas.scal(step, ds, n = dims['l'] + sum(dims['q']))
        blas.scal(step, dz, n = dims['l'] + sum(dims['q']))
        #print "scal 0 ds=\n", localmisc.strMat(ds), "\ndz=\n", localmisc.strMat(dz)

        ds[:dims['l']] += 1.0
        dz[:dims['l']] += 1.0
        ds[indq[:-1]] += 1.0
        dz[indq[:-1]] += 1.0
        #print "scal 1 ds=\n", localmisc.strMat(ds), "\ndz=\n", localmisc.strMat(dz)
        helpers.sp_create("update-dsdz", 7500)

        # ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
        #
        # This replaces the 'l' and 'q' components of ds and dz with the
        # updated variables in the current scaling.  
        # The 's' components of ds and dz are replaced with 
        #
        #     diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2} 
        #     diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2} 
        #
        helpers.sp_minor_push(7500)
        localmisc.scale2(lmbda, ds, dims, inverse = 'I')
        localmisc.scale2(lmbda, dz, dims, inverse = 'I')
        helpers.sp_minor_pop()
        #print "scale2 ds=\n", localmisc.strMat(ds), "\ndz=\n", localmisc.strMat(dz)

        # sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
        # sigz := ( e + step*sigz ) ./ lambda for 's' blocks.
        blas.scal(step, sigs)
        blas.scal(step, sigz)
        sigs += 1.0
        sigz += 1.0
        blas.tbsv(lmbda, sigs, n = sum(dims['s']), k = 0, ldA = 1, 
            offsetA = dims['l'] + sum(dims['q']))
        blas.tbsv(lmbda, sigz, n = sum(dims['s']), k = 0, ldA = 1, 
            offsetA = dims['l'] + sum(dims['q']))

        #print "sigs=\n", localmisc.strMat(sigs), "\nsigz=\n", localmisc.strMat(sigz)

        # dsk := Ls = dsk * sqrt(sigs).  
        # dzk := Lz = dzk * sqrt(sigz).
        ind2, ind3 = dims['l'] + sum(dims['q']), 0
        for k in xrange(len(dims['s'])):
            m = dims['s'][k]
            for i in xrange(m):
                blas.scal(math.sqrt(sigs[ind3+i]), ds, offset = ind2 + m*i,
                    n = m) 
                blas.scal(math.sqrt(sigz[ind3+i]), dz, offset = ind2 + m*i,
                    n = m)
            ind2 += m*m
            ind3 += m


        # Update lambda and scaling.

        helpers.sp_create("pre-update-scaling", 7700)

        misc.update_scaling(W, lmbda, ds, dz)

        helpers.sp_create("post-update-scaling", 7800)

        # For kappa, tau block: 
        #
        #     dg := sqrt( (kappa + step*dkappa) / (tau + step*dtau) ) 
        #         = dg * sqrt( (1 - step*tk) / (1 - step*tt) )
        #
        #     lmbda[-1] := sqrt((tau + step*dtau) * (kappa + step*dkappa))
        #                = lmbda[-1] * sqrt(( 1 - step*tt) * (1 - step*tk))

        #print "-- dg=%.9f, dgi=%.9f" % (dg, dgi)
        dg *= math.sqrt(1.0 - step*tk) / math.sqrt(1.0 - step*tt) 
        dgi = 1.0 / dg
        #print "step=%.17f tt=%.17f, tk=%.17f" % (step, tt, tk)
        #print "a = %.17f" %( math.sqrt(1.0 - step*tt) * math.sqrt(1.0 - step*tk) )
        lmbda[-1] *= math.sqrt(1.0 - step*tt) * math.sqrt(1.0 - step*tk) 
        #print "-- dg=%.9f, dgi=%.9f, lmbda[-1]=%.17f" % (dg, dgi, lmbda[-1])


        # Unscale s, z, tau, kappa (unscaled variables are used only to 
        # compute feasibility residuals).

        blas.copy(lmbda, s, n = dims['l'] + sum(dims['q']))
        ind = dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, s, offset = ind2)
            blas.copy(lmbda, s, offsetx = ind, offsety = ind2, n = m, 
                incy = m+1)
            ind += m
            ind2 += m*m
        localmisc.scale(s, W, trans = 'T')
        #print "unscaled s=\n", localmisc.strMat(s)

        blas.copy(lmbda, z, n = dims['l'] + sum(dims['q']))
        ind = dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, z, offset = ind2)
            blas.copy(lmbda, z, offsetx = ind, offsety = ind2, n = m, 
                    incy = m+1)
            ind += m
            ind2 += m*m
        localmisc.scale(z, W, inverse = 'I')
        #print "unscaled z=\n", localmisc.strMat(z)

        kappa, tau = lmbda[-1]/dgi, lmbda[-1]*dgi
        gap = ( blas.nrm2(lmbda, n = lmbda.size[0]-1) / tau )**2
        helpers.sp_create("end-of-loop", 8000)
        #print " ** kappa = %.10f, tau = %.10f, gap = %.10f" % (kappa, tau, gap)


def coneqp(P, q, G = None, h = None, dims = None, A = None, b = None,
    initvals = None, kktsolver = None, xnewcopy = None, xdot = None,
    xaxpy = None, xscal = None, ynewcopy = None, ydot = None, yaxpy = None,
    yscal = None):
    """
    """
    import math
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix

    STEP = 0.99
    EXPON = 3

    try: DEBUG = options['debug']
    except KeyError: DEBUG = False

    # Use Mehrotra correction or not.
    try: correction = options['use_correction']
    except KeyError: correction = True


    try: MAXITERS = options['maxiters']
    except KeyError: MAXITERS = 100
    else: 
        if type(MAXITERS) is not int or MAXITERS < 1: 
            raise ValueError("options['maxiters'] must be a positive "\
                "integer")

    try: ABSTOL = options['abstol']
    except KeyError: ABSTOL = 1e-7
    else: 
        if type(ABSTOL) is not float and type(ABSTOL) is not int: 
            raise ValueError("options['abstol'] must be a scalar")

    try: RELTOL = options['reltol']
    except KeyError: RELTOL = 1e-6
    else: 
        if type(RELTOL) is not float and type(RELTOL) is not int: 
            raise ValueError("options['reltol'] must be a scalar")

    if RELTOL <= 0.0 and ABSTOL <= 0.0 :
        raise ValueError("at least one of options['reltol'] and " \
            "options['abstol'] must be positive")

    try: FEASTOL = options['feastol']
    except KeyError: FEASTOL = 1e-7
    else: 
        if (type(FEASTOL) is not float and type(FEASTOL) is not int) or \
            FEASTOL <= 0.0:
            raise ValueError("options['feastol'] must be a positive "\
                "scalar")

    try: show_progress = options['show_progress']
    except KeyError: show_progress = True


    if kktsolver is None: 
        if dims and (dims['q'] or dims['s']):  
            kktsolver = 'chol'            
        else:
            kktsolver = 'chol2'            
    defaultsolvers = ('ldl', 'ldl2', 'chol', 'chol2')
    if type(kktsolver) is str and kktsolver not in defaultsolvers:
        raise ValueError("'%s' is not a valid value for kktsolver" \
            %kktsolver)


    # Argument error checking depends on level of customization.
    customkkt = type(kktsolver) is not str
    matrixP = type(P) in (matrix, spmatrix)
    matrixG = type(G) in (matrix, spmatrix)
    matrixA = type(A) in (matrix, spmatrix)
    if (not matrixP or (not matrixG and G is not None) or 
        (not matrixA and A is not None)) and not customkkt:
        raise ValueError("use of function valued P, G, A requires a "\
            "user-provided kktsolver")
    customx = (xnewcopy != None or xdot != None or xaxpy != None or 
        xscal != None) 
    if customx and (matrixP or matrixG or matrixA or not customkkt):
        raise ValueError("use of non-vector type for x requires "\
            "function valued P, G, A and user-provided kktsolver")
    customy = (ynewcopy != None or ydot != None or yaxpy != None or 
        yscal != None) 
    if customy and (matrixA or not customkkt):
        raise ValueError("use of non vector type for y requires "\
            "function valued A and user-provided kktsolver")


    if not customx and (type(q) is not matrix or q.typecode != 'd' or
        q.size[1] != 1):
        raise TypeError("'q' must be a 'd' matrix with one column")

    if matrixP:
        if P.typecode != 'd' or P.size != (q.size[0], q.size[0]):
            raise TypeError("'P' must be a 'd' matrix of size (%d, %d)"\
                %(q.size[0], q.size[0]))
        def fP(x, y, alpha = 1.0, beta = 0.0):
            base.symv(P, x, y, alpha = alpha, beta = beta)
    else:
        fP = P


    if h is None: h = matrix(0.0, (0,1))
    if type(h) is not matrix or h.typecode != 'd' or h.size[1] != 1:
        raise TypeError("'h' must be a 'd' matrix with one column")

    if not dims: dims = {'l': h.size[0], 'q': [], 's': []}
    if type(dims['l']) is not int or dims['l'] < 0: 
        raise TypeError("'dims['l']' must be a nonnegative integer")
    if [ k for k in dims['q'] if type(k) is not int or k < 1 ]:
        raise TypeError("'dims['q']' must be a list of positive integers")
    if [ k for k in dims['s'] if type(k) is not int or k < 0 ]:
        raise TypeError("'dims['s']' must be a list of nonnegative " \
            "integers")

    try: refinement = options['refinement']
    except KeyError: 
        if dims['q'] or dims['s']: refinement = 1
        else: refinement = 0
    else:
        if type(refinement) is not int or refinement < 0: 
            raise ValueError("options['refinement'] must be a "\
                "nonnegative integer")


    cdim = dims['l'] + sum(dims['q']) + sum([ k**2 for k in dims['s'] ])
    if h.size[0] != cdim:
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %cdim)

    # Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G.
    indq = [ dims['l'] ]  
    for k in dims['q']:  indq = indq + [ indq[-1] + k ] 

    # Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G.
    inds = [ indq[-1] ]
    for k in dims['s']:  inds = inds + [ inds[-1] + k**2 ] 

    if G is None:
        if customx:
            def G(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else:
            G = spmatrix([], [], [], (0, q.size[0]))
            matrixG = True
    if matrixG:
        if G.typecode != 'd' or G.size != (cdim, q.size[0]):
            raise TypeError("'G' must be a 'd' matrix of size (%d, %d)"\
                %(cdim, q.size[0]))
        def fG(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            misc.sgemv(G, x, y, dims, trans = trans, alpha = alpha, 
                beta = beta)
    else:
        fG = G


    if A is None:
        if customx or customy:
            def A(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else:
            A = spmatrix([], [], [], (0, q.size[0]))
            matrixA = True
    if matrixA:
        if A.typecode != 'd' or A.size[1] != q.size[0]:
            raise TypeError("'A' must be a 'd' matrix with %d columns" \
                %q.size[0])
        def fA(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            base.gemv(A, x, y, trans = trans, alpha = alpha, beta = beta)
    else:
        fA = A
    if not customy:
        if b is None: b = matrix(0.0, (0,1))
        if type(b) is not matrix or b.typecode != 'd' or b.size[1] != 1:
            raise TypeError("'b' must be a 'd' matrix with one column")
        if matrixA and b.size[0] != A.size[0]:
            raise TypeError("'b' must have length %d" %A.size[0])
    if b is None and customy:  
        raise ValueEror("use of non-vector type for y requires b")


    ws3, wz3 = matrix(0.0, (cdim,1 )), matrix(0.0, (cdim,1 ))
    def res(ux, uy, uz, us, vx, vy, vz, vs, W, lmbda):

        # Evaluates residual in Newton equations:
        # 
        #      [ vx ]    [ vx ]   [ 0     ]   [ P  A'  G' ]   [ ux        ]
        #      [ vy ] := [ vy ] - [ 0     ] - [ A  0   0  ] * [ uy        ]
        #      [ vz ]    [ vz ]   [ W'*us ]   [ G  0   0  ]   [ W^{-1}*uz ]
        #
        #      vs := vs - lmbda o (uz + us).

        # vx := vx - P*ux - A'*uy - G'*W^{-1}*uz
        fP(ux, vx, alpha = -1.0, beta = 1.0)
        fA(uy, vx, alpha = -1.0, beta = 1.0, trans = 'T') 
        blas.copy(uz, wz3)
        misc.scale(wz3, W, inverse = 'I')
        fG(wz3, vx, alpha = -1.0, beta = 1.0, trans = 'T') 

        # vy := vy - A*ux
        fA(ux, vy, alpha = -1.0, beta = 1.0)

        # vz := vz - G*ux - W'*us
        fG(ux, vz, alpha = -1.0, beta = 1.0)
        blas.copy(us, ws3)
        misc.scale(ws3, W, trans = 'T')
        blas.axpy(ws3, vz, alpha = -1.0)
 
        # vs := vs - lmbda o (uz + us)
        blas.copy(us, ws3)
        blas.axpy(uz, ws3)
        misc.sprod(ws3, lmbda, dims, diag = 'D')
        blas.axpy(ws3, vs, alpha = -1.0)


    # kktsolver(W) returns a routine for solving 
    #
    #     [ P   A'  G'*W^{-1} ] [ ux ]   [ bx ]
    #     [ A   0   0         ] [ uy ] = [ by ].
    #     [ G   0   -W'       ] [ uz ]   [ bz ]

    if kktsolver in defaultsolvers:
         if b.size[0] > q.size[0]:
             raise ValueError("Rank(A) < p or Rank([P; G; A]) < n")
         if kktsolver == 'ldl': 
             factor = localmisc.kkt_ldl(G, dims, A)
         elif kktsolver == 'ldl2': 
             factor = misc.kkt_ldl2(G, dims, A)
         elif kktsolver == 'chol':
             factor = misc.kkt_chol(G, dims, A)
         else:
             factor = misc.kkt_chol2(G, dims, A)
         def kktsolver(W):
             return factor(W, P)

    if xnewcopy is None: xnewcopy = matrix 
    if xdot is None: xdot = blas.dot
    if xaxpy is None: xaxpy = blas.axpy 
    if xscal is None: xscal = blas.scal 
    def xcopy(x, y): 
        xscal(0.0, y) 
        xaxpy(x, y)
    if ynewcopy is None: ynewcopy = matrix 
    if ydot is None: ydot = blas.dot 
    if yaxpy is None: yaxpy = blas.axpy 
    if yscal is None: yscal = blas.scal
    def ycopy(x, y): 
        yscal(0.0, y) 
        yaxpy(x, y)

    resx0 = max(1.0, math.sqrt(xdot(q,q)))
    resy0 = max(1.0, math.sqrt(ydot(b,b)))
    resz0 = max(1.0, misc.snrm2(h, dims))
    print "resx0: %.17f, resy0: %.17f, resz0: %.17f" %( resx0, resy0, resz0)

    if cdim == 0: 

        # Solve
        #
        #     [ P  A' ] [ x ]   [ -q ]
        #     [       ] [   ] = [    ].
        #     [ A  0  ] [ y ]   [  b ]

        try: f3 = kktsolver({'d': matrix(0.0, (0,1)), 'di': 
            matrix(0.0, (0,1)), 'beta': [], 'v': [], 'r': [], 'rti': []})
        except ArithmeticError: 
            raise ValueError("Rank(A) < p or Rank([P; A; G]) < n")
        x = xnewcopy(q)  
        xscal(-1.0, x)
        y = ynewcopy(b)
        f3(x, y, matrix(0.0, (0,1)))

        # dres = || P*x + q + A'*y || / resx0 
        rx = xnewcopy(q)
        fP(x, rx, beta = 1.0)
        pcost = 0.5 * (xdot(x, rx) + xdot(x, q))
        fA(y, rx, beta = 1.0, trans = 'T')
        dres = math.sqrt(xdot(rx, rx)) / resx0

        # pres = || A*x - b || / resy0
        ry = ynewcopy(b)
        fA(x, ry, alpha = 1.0, beta = -1.0)
        pres = math.sqrt(ydot(ry, ry)) / resy0 

        if pcost == 0.0: relgap = None
        else: relgap = 0.0

        return { 'status': 'optimal', 'x': x,  'y': y, 'z': 
            matrix(0.0, (0,1)), 's': matrix(0.0, (0,1)), 
            'gap': 0.0, 'relgap': 0.0, 
            'primal objective': pcost,
            'dual objective': pcost,
            'primal slack': 0.0, 'dual slack': 0.0,
            'primal infeasibility': pres, 'dual infeasibility': dres,
            'iterations': 0 } 


    x, y = xnewcopy(q), ynewcopy(b)  
    s, z = matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1))

    if initvals is None:

        # Factor
        #
        #     [ P   A'  G' ] 
        #     [ A   0   0  ].
        #     [ G   0  -I  ]
        
        W = {}
        W['d'] = matrix(1.0, (dims['l'], 1)) 
        W['di'] = matrix(1.0, (dims['l'], 1)) 
        W['v'] = [ matrix(0.0, (m,1)) for m in dims['q'] ]
        W['beta'] = len(dims['q']) * [ 1.0 ] 
        for v in W['v']: v[0] = 1.0
        W['r'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
        W['rti'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
        for r in W['r']: r[::r.size[0]+1 ] = 1.0
        for rti in W['rti']: rti[::rti.size[0]+1 ] = 1.0
        try: f = kktsolver(W)
        except ArithmeticError:  
            raise ValueError("Rank(A) < p or Rank([P; A; G]) < n")

             
        # Solve
        #
        #     [ P   A'  G' ]   [ x ]   [ -q ]
        #     [ A   0   0  ] * [ y ] = [  b ].
        #     [ G   0  -I  ]   [ z ]   [  h ]

        xcopy(q, x)
        xscal(-1.0, x)
        ycopy(b, y)  
        blas.copy(h, z)
        try: f(x, y, z) 
        except ArithmeticError:  
            raise ValueError("Rank(A) < p or Rank([P; G; A]) < n")
        blas.copy(z, s)  
        blas.scal(-1.0, s)  

        nrms = misc.snrm2(s, dims)
        ts = misc.max_step(s, dims)
        if ts >= -1e-8 * max(nrms, 1.0):  
            a = 1.0 + ts  
            s[:dims['l']] += a
            s[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                s[ind : ind+m*m : m+1] += a
                ind += m**2

        nrmz = misc.snrm2(z, dims)
        tz = misc.max_step(z, dims)
        if tz >= -1e-8 * max(nrmz, 1.0):
            a = 1.0 + tz  
            z[:dims['l']] += a
            z[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                z[ind : ind+m*m : m+1] += a
                ind += m**2


    else: 

        if 'x' in initvals: 
            xcopy(initvals['x'], x)
        else: 
            xscal(0.0, x)

        if 's' in initvals:
            blas.copy(initvals['s'], s)
            # ts = min{ t | s + t*e >= 0 }
            if misc.max_step(s, dims) >= 0:
                raise ValueError("initial s is not positive")
        else: 
            s[: dims['l']] = 1.0 
            ind = dims['l']
            for m in dims['q']:
                s[ind] = 1.0
                ind += m
            for m in dims['s']:
                s[ind : ind + m*m : m+1] = 1.0
                ind += m**2

        if 'y' in initvals:
            ycopy(initvals['y'], y)
        else:
            yscal(0.0, y)

        if 'z' in initvals:
            blas.copy(initvals['z'], z)
            # tz = min{ t | z + t*e >= 0 }
            if misc.max_step(z, dims) >= 0:
                raise ValueError("initial z is not positive")
        else:
            z[: dims['l']] = 1.0 
            ind = dims['l']
            for m in dims['q']:
                z[ind] = 1.0
                ind += m
            for m in dims['s']:
                z[ind : ind + m*m : m+1] = 1.0
                ind += m**2


    rx, ry, rz = xnewcopy(q), ynewcopy(b), matrix(0.0, (cdim, 1)) 
    dx, dy = xnewcopy(x), ynewcopy(y)   
    dz, ds = matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1))
    lmbda = matrix(0.0, (dims['l'] + sum(dims['q']) + sum(dims['s']), 1))
    lmbdasq = matrix(0.0, (dims['l'] + sum(dims['q']) + sum(dims['s']), 1))
    sigs = matrix(0.0, (sum(dims['s']), 1))
    sigz = matrix(0.0, (sum(dims['s']), 1))


    if show_progress: 
        print("% 10s% 12s% 10s% 8s% 7s" %("pcost", "dcost", "gap", "pres",
            "dres"))

    gap = misc.sdot(s, z, dims) 


    for iters in xrange(MAXITERS + 1):

        # f0 = (1/2)*x'*P*x + q'*x + r and  rx = P*x + q + A'*y + G'*z.
        xcopy(q, rx)
        fP(x, rx, beta = 1.0)
        f0 = 0.5 * (xdot(x, rx) + xdot(x, q))
        fA(y, rx, beta = 1.0, trans = 'T')
        fG(z, rx, beta = 1.0, trans = 'T')
        resx = math.sqrt(xdot(rx, rx))
           
        # ry = A*x - b
        ycopy(b, ry)
        fA(x, ry, alpha = 1.0, beta = -1.0)
        resy = math.sqrt(ydot(ry, ry))

        # rz = s + G*x - h
        blas.copy(s, rz)
        blas.axpy(h, rz, alpha = -1.0)
        fG(x, rz, beta = 1.0)
        resz = misc.snrm2(rz, dims)


        # Statistics for stopping criteria.

        # pcost = (1/2)*x'*P*x + q'*x 
        # dcost = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h)
        #       = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h+s) - z'*s
        #       = (1/2)*x'*P*x + q'*x + y'*ry + z'*rz - gap
        #print "resx: %.17f, resy: %.17f, resz: %.17f" %( resx, resy, resz)
        pcost = f0
        dcost = f0 + ydot(y, ry) + misc.sdot(z, rz, dims) - gap
        if pcost < 0.0:
            relgap = gap / -pcost
        elif dcost > 0.0:
            relgap = gap / dcost 
        else:
            relgap = None
        pres = max(resy/resy0, resz/resz0)
        dres = resx/resx0 

        if show_progress:
            print("%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e" \
                %(iters, pcost, dcost, gap, pres, dres))

        if ( pres <= FEASTOL and dres <= FEASTOL and ( gap <= ABSTOL or 
            (relgap is not None and relgap <= RELTOL) )) or \
            iters == MAXITERS:
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(s, m, ind)
                misc.symm(z, m, ind)
                ind += m**2
            ts = misc.max_step(s, dims)
            tz = misc.max_step(z, dims)
            if iters == MAXITERS:
                if show_progress:
                    print("Terminated (maximum number of iterations "\
                        "reached).")
                status = 'unknown'
            else:
                if show_progress:
                    print("Optimal solution found.")
                status = 'optimal'
            return { 'x': x,  'y': y,  's': s,  'z': z,  'status': status,
                    'gap': gap,  'relative gap': relgap, 
                    'primal objective': pcost,  'dual objective': dcost,
                    'primal infeasibility': pres,
                    'dual infeasibility': dres, 'primal slack': -ts,
                    'dual slack': -tz , 'iterations': iters }
                    

        # Compute initial scaling W and scaled iterates:  
        #
        #     W * z = W^{-T} * s = lambda.
        # 
        # lmbdasq = lambda o lambda.
        
        if iters == 0:
            W = misc.compute_scaling(s, z, lmbda, dims)
            #print "-- initial lmbda=\n", localmisc.strMat(lmbda)
        misc.ssqr(lmbdasq, lmbda, dims)


        # f3(x, y, z) solves
        #
        #    [ P   A'  G'    ] [ ux        ]   [ bx ]
        #    [ A   0   0     ] [ uy        ] = [ by ].
        #    [ G   0   -W'*W ] [ W^{-1}*uz ]   [ bz ]
        #
        # On entry, x, y, z containg bx, by, bz.
        # On exit, they contain ux, uy, uz.

        try: f3 = kktsolver(W)
        except ArithmeticError: 
            if iters == 0:
                raise ValueError("Rank(A) < p or Rank([P; A; G]) < n")
            else:  
                ind = dims['l'] + sum(dims['q'])
                for m in dims['s']:
                    misc.symm(s, m, ind)
                    misc.symm(z, m, ind)
                    ind += m**2
                ts = misc.max_step(s, dims)
                tz = misc.max_step(z, dims)
                print("Terminated (singular KKT matrix).")
                return { 'x': x,  'y': y,  's': s,  'z': z,  
                    'status': 'unknown', 'gap': gap,  
                    'relative gap': relgap, 'primal objective': pcost,  
                    'dual objective': dcost, 'primal infeasibility': pres,
                    'dual infeasibility': dres, 'primal slack': -ts,
                    'dual slack': -tz, 'iterations': iters }   

        # f4_no_ir(x, y, z, s) solves
        # 
        #     [ 0     ]   [ P  A'  G' ]   [ ux        ]   [ bx ]
        #     [ 0     ] + [ A  0   0  ] * [ uy        ] = [ by ]
        #     [ W'*us ]   [ G  0   0  ]   [ W^{-1}*uz ]   [ bz ]
        #
        #     lmbda o (uz + us) = bs.
        #
        # On entry, x, y, z, s contain bx, by, bz, bs.
        # On exit, they contain ux, uy, uz, us.

        def f4_no_ir(x, y, z, s):

            # Solve 
            #
            #     [ P A' G'   ] [ ux        ]    [ bx                    ]
            #     [ A 0  0    ] [ uy        ] =  [ by                    ]
            #     [ G 0 -W'*W ] [ W^{-1}*uz ]    [ bz - W'*(lmbda o\ bs) ]
            #
            #     us = lmbda o\ bs - uz.
            #
            # On entry, x, y, z, s  contains bx, by, bz, bs. 
            # On exit they contain x, y, z, s.
            
            # s := lmbda o\ s 
            #    = lmbda o\ bs
            misc.sinv(s, lmbda, dims)

            # z := z - W'*s 
            #    = bz - W'*(lambda o\ bs)
            blas.copy(s, ws3)
            misc.scale(ws3, W, trans = 'T')
            blas.axpy(ws3, z, alpha = -1.0)

            # Solve for ux, uy, uz
            f3(x, y, z)

            # s := s - z 
            #    = lambda o\ bs - uz.
            blas.axpy(z, s, alpha = -1.0)


        # f4(x, y, z, s) solves the same system as f4_no_ir, but applies
        # iterative refinement.

        if iters == 0:
            if refinement or DEBUG:
                wx, wy = xnewcopy(q), ynewcopy(b) 
                wz, ws = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1)) 
            if refinement:
                wx2, wy2 = xnewcopy(q), ynewcopy(b) 
                wz2, ws2 = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1)) 

        def f4(x, y, z, s):
            if refinement or DEBUG: 
                xcopy(x, wx)        
                ycopy(y, wy)        
                blas.copy(z, wz)        
                blas.copy(s, ws)        
            f4_no_ir(x, y, z, s)        
            for i in xrange(refinement):
                xcopy(wx, wx2)        
                ycopy(wy, wy2)        
                blas.copy(wz, wz2)        
                blas.copy(ws, ws2)        
                res(x, y, z, s, wx2, wy2, wz2, ws2, W, lmbda) 
                f4_no_ir(wx2, wy2, wz2, ws2)
                xaxpy(wx2, x)
                yaxpy(wy2, y)
                blas.axpy(wz2, z)
                blas.axpy(ws2, s)
            if DEBUG:
                res(x, y, z, s, wx, wy, wz, ws, W, lmbda)
                print("KKT residuals:")
                print("    'x': %e" %math.sqrt(xdot(wx, wx)))
                print("    'y': %e" %math.sqrt(ydot(wy, wy)))
                print("    'z': %e" %misc.snrm2(wz, dims))
                print("    's': %e" %misc.snrm2(ws, dims))


        mu = gap / (dims['l'] + len(dims['q']) + sum(dims['s']))
        sigma, eta = 0.0, 0.0

        for i in [0, 1]:

            # Solve
            #
            #     [ 0     ]   [ P  A' G' ]   [ dx        ]
            #     [ 0     ] + [ A  0  0  ] * [ dy        ] = -(1 - eta) * r
            #     [ W'*ds ]   [ G  0  0  ]   [ W^{-1}*dz ]
            #
            #     lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e (i=0)
            #     lmbda o (dz + ds) = -lmbda o lmbda - dsa o dza 
            #                         + sigma*mu*e (i=1) where dsa, dza
            #                         are the solution for i=0. 
 
            # ds = -lmbdasq + sigma * mu * e  (if i is 0)
            #    = -lmbdasq - dsa o dza + sigma * mu * e  (if i is 1), 
            #     where ds, dz are solution for i is 0.
            blas.scal(0.0, ds)
            if correction and i == 1:  
                blas.axpy(ws3, ds, alpha = -1.0)
            blas.axpy(lmbdasq, ds, n = dims['l'] + sum(dims['q']), 
                alpha = -1.0)
            ds[:dims['l']] += sigma*mu
            ind = dims['l']
            for m in dims['q']:
                ds[ind] += sigma*mu
                ind += m
            ind2 = ind
            for m in dims['s']:
                blas.axpy(lmbdasq, ds, n = m, offsetx = ind2, offsety =  
                    ind, incy = m + 1, alpha = -1.0)
                ds[ind : ind + m*m : m+1] += sigma*mu
                ind += m*m
                ind2 += m

       
            # (dx, dy, dz) := -(1 - eta) * (rx, ry, rz)
            xscal(0.0, dx);  xaxpy(rx, dx, alpha = -1.0 + eta)
            yscal(0.0, dy);  yaxpy(ry, dy, alpha = -1.0 + eta)
            blas.scal(0.0, dz) 
            blas.axpy(rz, dz, alpha = -1.0 + eta)

            try: f4(dx, dy, dz, ds)
            except ArithmeticError: 
                if iters == 0:
                    raise ValueError("Rank(A) < p or Rank([P; A; G]) < n")
                else:
                    ind = dims['l'] + sum(dims['q'])
                    for m in dims['s']:
                        misc.symm(s, m, ind)
                        misc.symm(z, m, ind)
                        ind += m**2
                    ts = misc.max_step(s, dims)
                    tz = misc.max_step(z, dims)
                    print("Terminated (singular KKT matrix).")
                    return { 'x': x,  'y': y,  's': s,  'z': z,  
                        'status': 'unknown', 'gap': gap,  
                        'relative gap': relgap, 'primal objective': pcost, 
                        'dual objective': dcost,
                        'primal infeasibility': pres,
                        'dual infeasibility': dres, 'primal slack': -ts,
                        'dual slack': -tz, 'iterations': iters }

            dsdz = misc.sdot(ds, dz, dims)

            # Save ds o dz for Mehrotra correction
            if correction and i == 0:
                blas.copy(ds, ws3)
                misc.sprod(ws3, dz, dims)

            # Maximum steps to boundary.  
            # 
            # If i is 1, also compute eigenvalue decomposition of the 
            # 's' blocks in ds,dz.  The eigenvectors Qs, Qz are stored in 
            # dsk, dzk.  The eigenvalues are stored in sigs, sigz.

            misc.scale2(lmbda, ds, dims)
            misc.scale2(lmbda, dz, dims)

            if i == 0: 
                ts = misc.max_step(ds, dims)
                tz = misc.max_step(dz, dims)
            else:
                ts = misc.max_step(ds, dims, sigma = sigs)
                tz = misc.max_step(dz, dims, sigma = sigz)
            t = max([ 0.0, ts, tz ])
            #print "== t=%.17f from " % t, str([ts, tz])
            if t == 0:
                step = 1.0
            else:
                if i == 0:
                    step = min(1.0, 1.0 / t)
                else:
                    step = min(1.0, STEP / t)
            if i == 0: 
                sigma = min(1.0, max(0.0, 
                    1.0 - step + dsdz/gap * step**2))**EXPON
                eta = 0.0
            #print "== step=%.17f sigma=%.17f dsdz=%.17f" %( step, sigma, dsdz)


        xaxpy(dx, x, alpha = step)
        yaxpy(dy, y, alpha = step)

        # We will now replace the 'l' and 'q' blocks of ds and dz with 
        # the updated iterates in the current scaling.
        # We also replace the 's' blocks of ds and dz with the factors 
        # Ls, Lz in a factorization Ls*Ls', Lz*Lz' of the updated variables
        # in the current scaling.

        # ds := e + step*ds for nonlinear, 'l' and 'q' blocks.
        # dz := e + step*dz for nonlinear, 'l' and 'q' blocks.
        blas.scal(step, ds, n = dims['l'] + sum(dims['q']))
        blas.scal(step, dz, n = dims['l'] + sum(dims['q']))
        ind = dims['l']
        ds[:ind] += 1.0
        dz[:ind] += 1.0
        for m in dims['q']:
            ds[ind] += 1.0
            dz[ind] += 1.0
            ind += m


        # ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
        #
        # This replaced the 'l' and 'q' components of ds and dz with the
        # updated iterates in the current scaling.
        # The 's' components of ds and dz are replaced with
        #
        #     diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2}
        #     diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2}
        # 
        misc.scale2(lmbda, ds, dims, inverse = 'I')
        misc.scale2(lmbda, dz, dims, inverse = 'I')

        # sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
        # sigz := ( e + step*sigz ) ./ lmabda for 's' blocks.
        blas.scal(step, sigs)
        blas.scal(step, sigz)
        sigs += 1.0
        sigz += 1.0
        blas.tbsv(lmbda, sigs, n = sum(dims['s']), k = 0, ldA = 1, offsetA
            = dims['l'] + sum(dims['q']))
        blas.tbsv(lmbda, sigz, n = sum(dims['s']), k = 0, ldA = 1, offsetA
            = dims['l'] + sum(dims['q']))

        # dsk := Ls = dsk * sqrt(sigs).
        # dzk := Lz = dzk * sqrt(sigz).
        ind2, ind3 = dims['l'] + sum(dims['q']), 0
        for k in xrange(len(dims['s'])):
            m = dims['s'][k]
            for i in xrange(m):
                blas.scal(math.sqrt(sigs[ind3+i]), ds, offset = ind2 + m*i,
                    n = m)
                blas.scal(math.sqrt(sigz[ind3+i]), dz, offset = ind2 + m*i,
                    n = m)
            ind2 += m*m
            ind3 += m


        # Update lambda and scaling.
        misc.update_scaling(W, lmbda, ds, dz)

        # Unscale s, z (unscaled variables are used only to compute 
        # feasibility residuals).

        blas.copy(lmbda, s, n = dims['l'] + sum(dims['q']))
        ind = dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, s, offset = ind2)
            blas.copy(lmbda, s, offsetx = ind, offsety = ind2, n = m, 
                incy = m+1)
            ind += m
            ind2 += m*m
        misc.scale(s, W, trans = 'T')

        blas.copy(lmbda, z, n = dims['l'] + sum(dims['q']))
        ind = dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, z, offset = ind2)
            blas.copy(lmbda, z, offsetx = ind, offsety = ind2, n = m, 
                incy = m+1)
            ind += m
            ind2 += m*m
        misc.scale(z, W, inverse = 'I')

        gap = blas.dot(lmbda, lmbda) 
        #print "== gap = %.17f" % gap




def lp(c, G, h, A = None, b = None, solver = None, primalstart = None,
    dualstart = None):

    import math
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix

    if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1: 
        raise TypeError("'c' must be a dense column matrix")
    n = c.size[0]
    if n < 1: raise ValueError("number of variables must be at least 1")

    if (type(G) is not matrix and type(G) is not spmatrix) or \
        G.typecode != 'd' or G.size[1] != n:
        raise TypeError("'G' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    m = G.size[0]
    if type(h) is not matrix or h.typecode != 'd' or h.size != (m,1):
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %m)

    if A is None:  A = spmatrix([], [], [], (0,n), 'd')
    if (type(A) is not matrix and type(A) is not spmatrix) or \
        A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)


    return conelp(c, G, h, {'l': m, 'q': [], 's': []}, A,  b, primalstart,
        dualstart)





def socp(c, Gl = None, hl = None, Gq = None, hq = None, A = None, b = None,
    solver = None, primalstart = None, dualstart = None):

    from cvxopt import base, blas
    from cvxopt.base import matrix, spmatrix

    if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1: 
        raise TypeError("'c' must be a dense column matrix")
    n = c.size[0]
    if n < 1: raise ValueError("number of variables must be at least 1")

    if Gl is None:  Gl = spmatrix([], [], [], (0,n), tc='d')
    if (type(Gl) is not matrix and type(Gl) is not spmatrix) or \
        Gl.typecode != 'd' or Gl.size[1] != n:
        raise TypeError("'Gl' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    ml = Gl.size[0]
    if hl is None: hl = matrix(0.0, (0,1))
    if type(hl) is not matrix or hl.typecode != 'd' or \
        hl.size != (ml,1):
        raise TypeError("'hl' must be a dense 'd' matrix of " \
            "size (%d,1)" %ml)

    if Gq is None: Gq = []
    if type(Gq) is not list or [ G for G in Gq if (type(G) is not matrix 
        and type(G) is not spmatrix) or G.typecode != 'd' or 
        G.size[1] != n ]:
        raise TypeError("'Gq' must be a list of sparse or dense 'd' "\
            "matrices with %d columns" %n)
    mq = [ G.size[0] for G in Gq ]
    a = [ k for k in xrange(len(mq)) if mq[k] == 0 ] 
    if a: raise TypeError("the number of rows of Gq[%d] is zero" %a[0])
    if hq is None: hq = []
    if type(hq) is not list or len(hq) != len(mq) or [ h for h in hq if
        (type(h) is not matrix and type(h) is not spmatrix) or 
        h.typecode != 'd' ]: 
        raise TypeError("'hq' must be a list of %d dense or sparse "\
            "'d' matrices" %len(mq))
    a = [ k for k in xrange(len(mq)) if hq[k].size != (mq[k], 1) ]
    if a:
        k = a[0]
        raise TypeError("'hq[%d]' has size (%d,%d).  Expected size "\
            "is (%d,1)." %(k, hq[k].size[0], hq[k].size[1], mq[k]))

    if A is None: A = spmatrix([], [], [], (0,n), 'd')
    if (type(A) is not matrix and type(A) is not spmatrix) or \
        A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)

    dims = {'l': ml, 'q': mq, 's': []}
    N = ml + sum(mq)


    h = matrix(0.0, (N,1))
    if type(Gl) is matrix or [ Gk for Gk in Gq if type(Gk) is matrix ]:
        G = matrix(0.0, (N, n))
    else:
        G = spmatrix([], [], [], (N, n), 'd')
    h[:ml] = hl
    G[:ml,:] = Gl
    ind = ml
    for k in xrange(len(mq)):
        h[ind : ind + mq[k]] = hq[k]
        G[ind : ind + mq[k], :] = Gq[k]
        ind += mq[k]

    if primalstart:
        ps = {}
        ps['x'] = primalstart['x']
        ps['s'] = matrix(0.0, (N,1))
        if ml: ps['s'][:ml] = primalstart['sl']
        if mq:
            ind = ml
            for k in xrange(len(mq)): 
                ps['s'][ind : ind + mq[k]] = primalstart['sq'][k][:]
                ind += mq[k]
    else: 
        ps = None

    if dualstart:
        ds = {}
        if p:  ds['y'] = dualstart['y']
        ds['z'] = matrix(0.0, (N,1))
        if ml: ds['z'][:ml] = dualstart['zl']
        if mq: 
            ind = ml
            for k in xrange(len(mq)):
                ds['z'][ind : ind + mq[k]] = dualstart['zq'][k][:]
                ind += mq[k]
    else: 
        ds = None

    sol = conelp(c, G, h, dims, A = A, b = b, primalstart = ps, dualstart
        = ds)
    if sol['s'] is None:  
        sol['sl'] = None
        sol['sq'] = None
    else: 
        sol['sl'] = sol['s'][:ml]  
        sol['sq'] = [ matrix(0.0, (m,1)) for m in mq ] 
        ind = ml
        for k in xrange(len(mq)):
            sol['sq'][k][:] = sol['s'][ind : ind+mq[k]]
            ind += mq[k]
    del sol['s']

    if sol['z'] is None: 
        sol['zl'] = None
        sol['zq'] = None
    else: 
        sol['zl'] = sol['z'][:ml]
        sol['zq'] = [ matrix(0.0, (m,1)) for m in mq] 
        ind = ml
        for k in xrange(len(mq)):
            sol['zq'][k][:] = sol['z'][ind : ind+mq[k]]
            ind += mq[k]
    del sol['z']

    return sol

    


def qp(P, q, G = None, h = None, A = None, b = None, solver = None, 
    initvals = None):
    from cvxopt import base, blas
    from cvxopt.base import matrix, spmatrix
    return coneqp(P, q, G, h, None, A,  b, initvals, kktsolver='ldl')




def lp(c, G, h, A = None, b = None, solver = None, primalstart = None,
    dualstart = None):

    import math
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix

    if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1: 
        raise TypeError("'c' must be a dense column matrix")
    n = c.size[0]
    if n < 1: raise ValueError("number of variables must be at least 1")

    if (type(G) is not matrix and type(G) is not spmatrix) or \
        G.typecode != 'd' or G.size[1] != n:
        raise TypeError("'G' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    m = G.size[0]
    if type(h) is not matrix or h.typecode != 'd' or h.size != (m,1):
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %m)

    if A is None:  A = spmatrix([], [], [], (0,n), 'd')
    if (type(A) is not matrix and type(A) is not spmatrix) or \
        A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)


    return conelp(c, G, h, {'l': m, 'q': [], 's': []}, A,  b, primalstart,
        dualstart)


def socp(c, Gl = None, hl = None, Gq = None, hq = None, A = None, b = None,
    solver = None, primalstart = None, dualstart = None):


    from cvxopt import base, blas
    from cvxopt.base import matrix, spmatrix

    if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1: 
        raise TypeError("'c' must be a dense column matrix")
    n = c.size[0]
    if n < 1: raise ValueError("number of variables must be at least 1")

    if Gl is None:  Gl = spmatrix([], [], [], (0,n), tc='d')
    if (type(Gl) is not matrix and type(Gl) is not spmatrix) or \
        Gl.typecode != 'd' or Gl.size[1] != n:
        raise TypeError("'Gl' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    ml = Gl.size[0]
    if hl is None: hl = matrix(0.0, (0,1))
    if type(hl) is not matrix or hl.typecode != 'd' or \
        hl.size != (ml,1):
        raise TypeError("'hl' must be a dense 'd' matrix of " \
            "size (%d,1)" %ml)

    if Gq is None: Gq = []
    if type(Gq) is not list or [ G for G in Gq if (type(G) is not matrix 
        and type(G) is not spmatrix) or G.typecode != 'd' or 
        G.size[1] != n ]:
        raise TypeError("'Gq' must be a list of sparse or dense 'd' "\
            "matrices with %d columns" %n)
    mq = [ G.size[0] for G in Gq ]
    a = [ k for k in range(len(mq)) if mq[k] == 0 ] 
    if a: raise TypeError("the number of rows of Gq[%d] is zero" %a[0])
    if hq is None: hq = []
    if type(hq) is not list or len(hq) != len(mq) or [ h for h in hq if
        (type(h) is not matrix and type(h) is not spmatrix) or 
        h.typecode != 'd' ]: 
        raise TypeError("'hq' must be a list of %d dense or sparse "\
            "'d' matrices" %len(mq))
    a = [ k for k in range(len(mq)) if hq[k].size != (mq[k], 1) ]
    if a:
        k = a[0]
        raise TypeError("'hq[%d]' has size (%d,%d).  Expected size "\
            "is (%d,1)." %(k, hq[k].size[0], hq[k].size[1], mq[k]))

    if A is None: A = spmatrix([], [], [], (0,n), 'd')
    if (type(A) is not matrix and type(A) is not spmatrix) or \
        A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)

    dims = {'l': ml, 'q': mq, 's': []}
    N = ml + sum(mq)


    h = matrix(0.0, (N,1))
    if type(Gl) is matrix or [ Gk for Gk in Gq if type(Gk) is matrix ]:
        G = matrix(0.0, (N, n))
    else:
        G = spmatrix([], [], [], (N, n), 'd')
    h[:ml] = hl
    G[:ml,:] = Gl
    ind = ml
    for k in range(len(mq)):
        h[ind : ind + mq[k]] = hq[k]
        G[ind : ind + mq[k], :] = Gq[k]
        ind += mq[k]

    if primalstart:
        ps = {}
        ps['x'] = primalstart['x']
        ps['s'] = matrix(0.0, (N,1))
        if ml: ps['s'][:ml] = primalstart['sl']
        if mq:
            ind = ml
            for k in range(len(mq)): 
                ps['s'][ind : ind + mq[k]] = primalstart['sq'][k][:]
                ind += mq[k]
    else: 
        ps = None

    if dualstart:
        ds = {}
        if p:  ds['y'] = dualstart['y']
        ds['z'] = matrix(0.0, (N,1))
        if ml: ds['z'][:ml] = dualstart['zl']
        if mq: 
            ind = ml
            for k in range(len(mq)):
                ds['z'][ind : ind + mq[k]] = dualstart['zq'][k][:]
                ind += mq[k]
    else: 
        ds = None

    sol = conelp(c, G, h, dims, A = A, b = b, primalstart = ps, dualstart
        = ds)
    if sol['s'] is None:  
        sol['sl'] = None
        sol['sq'] = None
    else: 
        sol['sl'] = sol['s'][:ml]  
        sol['sq'] = [ matrix(0.0, (m,1)) for m in mq ] 
        ind = ml
        for k in range(len(mq)):
            sol['sq'][k][:] = sol['s'][ind : ind+mq[k]]
            ind += mq[k]
    del sol['s']

    if sol['z'] is None: 
        sol['zl'] = None
        sol['zq'] = None
    else: 
        sol['zl'] = sol['z'][:ml]
        sol['zq'] = [ matrix(0.0, (m,1)) for m in mq] 
        ind = ml
        for k in range(len(mq)):
            sol['zq'][k][:] = sol['z'][ind : ind+mq[k]]
            ind += mq[k]
    del sol['z']

    return sol

    
def sdp(c, Gl = None, hl = None, Gs = None, hs = None, A = None, b = None, 
    solver = None, primalstart = None, dualstart = None):

    #print "start localcones.sdp ...."

    import math
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix

    if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1: 
        raise TypeError("'c' must be a dense column matrix")
    n = c.size[0]
    if n < 1: raise ValueError("number of variables must be at least 1")

    if Gl is None: Gl = spmatrix([], [], [], (0,n), tc='d')
    if (type(Gl) is not matrix and type(Gl) is not spmatrix) or \
        Gl.typecode != 'd' or Gl.size[1] != n:
        raise TypeError("'Gl' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    ml = Gl.size[0]
    if hl is None: hl = matrix(0.0, (0,1))
    if type(hl) is not matrix or hl.typecode != 'd' or \
        hl.size != (ml,1):
        raise TypeError("'hl' must be a 'd' matrix of size (%d,1)" %ml)

    if Gs is None: Gs = []
    if type(Gs) is not list or [ G for G in Gs if (type(G) is not matrix 
        and type(G) is not spmatrix) or G.typecode != 'd' or 
        G.size[1] != n ]:
        raise TypeError("'Gs' must be a list of sparse or dense 'd' "\
            "matrices with %d columns" %n)
    ms = [ int(math.sqrt(G.size[0])) for G in Gs ]
    a = [ k for k in range(len(ms)) if ms[k]**2 != Gs[k].size[0] ]
    if a: raise TypeError("the squareroot of the number of rows in "\
        "'Gs[%d]' is not an integer" %k)
    if hs is None: hs = []
    if type(hs) is not list or len(hs) != len(ms) or [ h for h in hs if
        (type(h) is not matrix and type(h) is not spmatrix) or
        h.typecode != 'd' ]:
        raise TypeError("'hs' must be a list of %d dense or sparse "\
            "'d' matrices" %len(ms))
    a = [ k for k in range(len(ms)) if hs[k].size != (ms[k],ms[k]) ]
    if a:
        k = a[0]
        raise TypeError("hs[%d] has size (%d,%d).  Expected size is "\
            "(%d,%d)." %(k,hs[k].size[0], hs[k].size[1], ms[k], ms[k]))

    if A is None: A = spmatrix([], [], [], (0,n), 'd')
    if (type(A) is not matrix and type(A) is not spmatrix) or \
        A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)

    dims = {'l': ml, 'q': [], 's': ms}
    N = ml + sum([ m**2 for m in ms ])

         
    h = matrix(0.0, (N,1))
    if type(Gl) is matrix or [ Gk for Gk in Gs if type(Gk) is matrix ]:
        G = matrix(0.0, (N, n))
    else:
        G = spmatrix([], [], [], (N, n), 'd')
    h[:ml] = hl
    G[:ml,:] = Gl
    ind = ml
    for k in range(len(ms)):
        m = ms[k]
        h[ind : ind + m*m] = hs[k][:]
        G[ind : ind + m*m, :] = Gs[k]
        ind += m**2

    if primalstart:
        ps = {}
        ps['x'] = primalstart['x']
        ps['s'] = matrix(0.0, (N,1))
        if ml: ps['s'][:ml] = primalstart['sl']
        if ms:
            ind = ml
            for k in range(len(ms)):
                m = ms[k]
                ps['s'][ind : ind + m*m] = primalstart['ss'][k][:]
                ind += m**2
    else: 
        ps = None

    if dualstart:
        ds = {}
        if p:  ds['y'] = dualstart['y']
        ds['z'] = matrix(0.0, (N,1))
        if ml: ds['z'][:ml] = dualstart['zl']
        if ms: 
            ind = ml
            for k in range(len(ms)):
                m = ms[k]
                ds['z'][ind : ind + m*m] = dualstart['zs'][k][:]
                ind += m**2
    else: 
        ds = None

    #print "** h=\n", helpers.str2(h, "%.3f")
    #print "** G=\n", helpers.str2(G, "%.3f")

    sol = conelp(c, G, h, dims, A=A, b=b, primalstart=ps, dualstart=ds, kktsolver='ldl')
    if sol['s'] is None:
        sol['sl'] = None
        sol['ss'] = None
    else:
        sol['sl'] = sol['s'][:ml]
        sol['ss'] = [ matrix(0.0, (mk, mk)) for mk in ms ]
        ind = ml
        for k in range(len(ms)):
            m = ms[k]
            sol['ss'][k][:] = sol['s'][ind:ind+m*m]
            ind += m**2
    del sol['s']

    if sol['z'] is None:
        sol['zl'] = None
        sol['zs'] = None
    else:
        sol['zl'] = sol['z'][:ml]
        sol['zs'] = [ matrix(0.0, (mk, mk)) for mk in ms ]
        ind = ml
        for k in range(len(ms)):
            m = ms[k]
            sol['zs'][k][:] = sol['z'][ind:ind+m*m]
            ind += m**2
    del sol['z']

    return sol


#def qp(P, q, G = None, h = None, A = None, b = None, solver = None, 
#    initvals = None):
#
#    return coneqp(P, q, G, h, None, A,  b, initvals)
