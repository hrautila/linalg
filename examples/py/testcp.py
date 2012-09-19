# The analytic centering with cone constraints example of section 9.1 
# (Problems with nonlinear objectives).

from cvxopt import matrix, log, div, spdiag 
#from cvxopt import solvers  
import localcvx, helpers
 
def F(x = None, z = None):  
     if x is None:
          return 0, matrix(0.0, (3,1))  
     #print "x=\n", helpers.str2(x)
     if max(abs(x)) >= 1.0:
          return None  
     u = 1 - x**2  
     #print "u=\n", helpers.str2(u)
     val = -sum(log(u))  
     Df = div(2*x, u).T  
     if z is None:
          #print "val ", str(val)
          #print "Df\n", helpers.str2(Df)
          return val, Df  
     H = spdiag(2 * z[0] * div(1 + u**2, u**2))  
     #print "val ", str(val)
     #print "Df\n", helpers.str2(Df)
     #print "H\n", helpers.str2(H)
     return val, Df, H  
 
G = matrix([ 
    [0., -1.,  0.,  0., -21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],
    [0.,  0., -1.,  0.,   0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],
    [0.,  0.,  0., -1.,  -5.,   2., -17.,   2.,  -6.,   8., -17.,  -7., 6.]
    ])  
h = matrix(
    [1.0, 0.0, 0.0, 0.0, 20., 10., 40., 10., 80., 10., 40., 10., 15.])  
dims = {'l': 0, 'q': [4], 's':  [3]}  
helpers.sp_reset("./sp.cp")
helpers.sp_activate()
#localcvx.options['maxiters'] = 3
sol = localcvx.cp(F, G, h, dims)  
print("\nx = \n") 
print(sol['x'])
