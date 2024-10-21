# program to solve the coupled differential eqns
#  d^2phi/dr + 1/r dphi/dr            + mu**2 phi - 2 lambda phi**3 - kappa phi xi**2 = 0
#  d^2xi /dr + 1/r dxi/dr - 1/r**2 xi + mu'**2 xi - 2 lambda' xi**3 - kappa xi phi**2 = 0
# with  boundary conditions
#                  dphi(0) = 0, xi(0) = 0, phi(inf) = v, xi(inf) = v'


# First version
# 03 oct 2020 -
#    solve for kappa = 0 each eq. separately
#
# Second version
# 04 oct 2020 -- 05 oct 2020
#  solve for kappa /= 0, coupled eqns.

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib as mpl
import matplotlib.pyplot as plt
import math



#mpl.rcParams.update({'font.size': 15})


# DEFINE
# Y = {phi, phi'}  = {y0,y1}
# Y'= {phi',phi''} = {y1, f(phi,r)}

v     = 0.1
lmb   = 1.0   

def f(r,y):
    return np.vstack((y[1], -msq*y[0] + 2*lmb*(y[0])**3))

def bc(ya,yb):
    return np.array([ya[0],yb[0] - v])

S = [[0,0],[0,-1]]

msq  = 2*lmb * v**2 
# define mesh
r0 = 0
rf = 30
N = 10**5
r = np.linspace(r0,rf,N)

y = np.zeros((2,len(r)))
    
# initial guess
y[0,:] = 0.1  #phi
y[1,:] = 0 # dphi
#solve
res = solve_bvp(f,bc,r,y,S = S, max_nodes = 5000)
# plots 
y_plot  = res.sol(r)[0]
dy_plot = res.sol(r)[1]   
if res.status != 0:
    print("Warining: res.status is %d " % (res.status))

print(dy_plot[0])    
    
plt.plot(r,y_plot)
plt.xlabel('$r$',fontsize =16)
plt.ylabel('$\phi$',fontsize =16)
plt.show()
  
