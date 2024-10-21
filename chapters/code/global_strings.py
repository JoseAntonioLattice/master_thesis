import numpy as np
from scipy.integrate import solve_bvp
import scipy.special as spl
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

# phi''(r) + phi'(r)/r - n^2 phi(r)/r^2 - m^2 phi(r) - 4 \lambda phi^3(r) = 0

v = 1.0
lmb = 1.0
msq = -lmb*v**2

n = 1.0

def derivative(r,y):
    return np.vstack((y[1], -y[1]/r + y[0]*n**2/r**2 + msq*y[0] + lmb*(y[0])**3))

def energy_den(y,dy): 
    return 0.5 * dy**2 + 0.5 * msq * y**2 + 0.25*lmb*y**4 - 0.5 * msq * v**2 - 0.25*lmb*v**4


def bc(ya,yb):
    return np.array([ya[0],yb[0]-v])

r0 = 1e-5
rf = 20.0
N = 500
r = np.linspace(r0,rf,N)
y = np.zeros((2,N))
y_guess = np.zeros(N)
dy_guess = np.zeros(N)
e = np.zeros(N)
df = np.zeros(N)

for i in range (N):
    y_guess[i] = v*(-math.exp(-r[i])+1.0)
    dy_guess[i]= v*math.exp(-r[i])
    
y[0] = y_guess
y[1] = dy_guess
res = solve_bvp(derivative,bc,r,y)

y_plot = res.sol(r)[0]
dy_plot = res.sol(r)[1]


datafile = open("global_profile.dat",'w')

for i in range(N):
  e[i] = energy_den(y_plot[i],dy_plot[i])
  df[i] = v-n**2/(2*lmb*v*r[i]**2)
  print(r[i],y_plot[i],e[i], file = datafile)



fig = plt.figure(figsize=(8,5))
ax1 = fig.add_axes([0.10,0.10,0.70,0.85])

ax1.plot(v*np.sqrt(lmb)*r,y_plot,v*np.sqrt(lmb)*r,e)
#plt.ylim([-.3,1.2])
plt.text(8,0.9,'$f(\sqrt{v^2\\lambda} r)$',fontsize=16)
plt.text(8,-0.2,'$\epsilon$',fontsize=16)
plt.xlabel('$\sqrt{v^2\\lambda}r$')

plt.show()
