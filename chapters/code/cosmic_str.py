# Purpose:
#  Program for solving the system of coupled differential eqns
#  d_r^2 phi + 1/r dphi/dr - 1/r**2 ( n**2 + 2 n h a/r + h**2  a**2/r**2) phi - m**2 phi - lambda phi**3 - kappa phi xi**2 = 0
#  d_r^2 xi  + 1/r dxi/dr  - 1/r**2 (n'**2 + 2 n'h'a/r + h'**2 a**2/r**2) xi  - m'**2 xi - lambda' xi**3 - kappa xi phi**2 = 0
#  d_r^2 a - 1/r d_r a - h**2 n phi**2 - h**2 a phi**2 - h' n' xi**2 - h'**2 a xi**2 = 0 
# with  boundary conditions
#    dphi(0) = 0, xi(0) = 0, phi(inf) = v, xi(inf) = v', a(0) = 0, a(inf) = -n/h = -n'/h'
#
#
# First version
# 03 oct 2020 -
#    solve for kappa = 0 each eq. separately
#
# Second version
# 04 oct 2020 -- 05 oct 2020
#  solve for kappa /= 0, coupled eqns.
#
# Third version
# 28 feb 2021 -- 28 feb 2021
# add n and n' to the equations
#
# Fourth version
# 07 mar 2021 -- 09 mar 2021
# add gauge field A = a(r)/r e_{\varphi}
#
# Fifth version
# 10 mar 2021 -- 10 mar 2021
# Implement euclidian version
#
# 6th versiono
# 29 august 2021
# Compute energy density

import numpy as np
from scipy.integrate import solve_bvp
import scipy.special as spl
import matplotlib as mpl
import matplotlib.pyplot as plt
import math



#mpl.rcParams.update({'font.size': 15})
# DEFINE
# Y = {phi, phi' ,xi ,xi' ,a ,a' } = {y0,y1,y2,y3,y4,y5}
# Y'= {phi',phi'',xi',xi'',a',a''} = {y1, -1/r y1 + ( n**2 + 2 h n a/r + h**2  a**2/r**2)y0/r**2 - mu**2  y0 + 2 lambda  y0**3 + kappa y0 y2**2,
#                                     y3, -1/r y3 + (n'**2 + 2 h'n'a/r + h'**2 a**2/r**2)y2/r**2 - mu'**2 y2 + 2 lambda' y2**3 + kappa y2 y0**2,
#                                     y5,  1/r y5 + h n y0**2 + h**2 y4 y0**2 + h' n' y2**2 + h'**2 y4 y2**2}
#

v    = 0.5
vp   = 1.0
lmb  = 1.0  
lmbp = 1.0
n    = 0
npp  = -2
h    = 0.0
hp   = 1.0#npp*h/n
a_inf = -npp/hp
def der(r,y,kappa,msq,msqp):
    return np.vstack((y[1], -y[1]/r + y[0]/r**2*(  n**2 + 2*h*n*y[4]    +  h**2*(y[4])**2) +  msq*y[0] \
                            + lmb*(y[0])**3  + kappa*y[0]*(y[2])**2,\
                      y[3], -y[3]/r + y[2]/r**2*(npp**2 + 2*hp*npp*y[4] + hp**2*(y[4])**2) + msqp*y[2] \
                            + lmbp*(y[2])**3 + kappa*y[2]*(y[0])**2, \
                      y[5], y[5]/r  + h*n*(y[0])**2 + h**2*y[4]*(y[0])**2 + hp*npp*(y[2])**2 + hp**2*y[4]*(y[2])**2 ))

#def bc(ya,yb):
#    return np.array([ya[0],yb[0] - v,ya[2],yb[2] - vp,ya[4],yb[4] - a_inf])


def bc(ya,yb):
    return np.array([ya[1],yb[1],ya[2],yb[2]-vp,ya[4],yb[4] - a_inf])

#kmax = min(np.sqrt(lmb*lmbp),msqp*lmb/msq,msq*lmbp/msq)
                   
NN = 21
k0 = -1*np.sqrt(lmb*lmbp)+0.1
kf =  -k0


#kappaarray = np.array([kf,k0])
kappaarray = np.linspace(k0,kf,NN)
msqarray  = -kappaarray*vp**2 - lmb * v**2 
msqparray = -kappaarray* v**2 - lmbp*vp**2 

#NN = len(kappaarray)

# define mesh
r0 = 1e-3
rf = 50.0
N = 500

r = np.linspace(r0,rf,N)

yy = np.zeros((len(r),NN))
y = np.zeros((6,r.size))
y_guess = np.zeros(r.size)
y_guess = y[0,:]
yy2 = np.zeros((len(r),NN))
y2_guess = np.zeros(r.size)
yy3 = np.zeros((len(r),NN))
y3_guess = np.zeros(r.size)
dy_guess = np.zeros(r.size)
dy2_guess = np.zeros(r.size)
dy3_guess = np.zeros(r.size)
cmap = plt.get_cmap('jet',NN)
e = np.zeros((len(r),NN))


datafile = open("data/profile_l=%1.2f_lp=%1.2f_n=%d_np=%d_h=%1.2f_hp=%1.2f_v=%1.2f_vp=%1.2f.dat" % (lmb,lmbp,n,npp,h,hp,v,vp),'w')
datafile2 = open("consistency.dat",'w')


for jj in range(len(r)):
    y_guess[jj]  = v*(-math.exp(-r[jj]**2)+1)
    y2_guess[jj] =vp*(-math.exp(-r[jj]**2)+1) 
    y3_guess[jj] = a_inf*(1+math.exp(-r[jj]**2)) 
    dy_guess[jj] =  2* v*r[jj]*math.exp(-r[jj]**2)
    dy2_guess[jj] = 2*vp*r[jj]*math.exp(-r[jj]**2)# 1.2*2*r[jj]*math.exp(-r[jj]*r[jj])#
    dy3_guess[jj] = -2*r[jj]*a_inf*math.exp(-r[jj]**2)#
    
for ii in range(NN):
    def dd(r,y):
        return der(r,y,kappaarray[ii],msqarray[ii],msqparray[ii])
    # initial guess
    y[0,:] = y_guess   #  phi
    y[1,:] = dy_guess  # dphi
    y[2,:] = y2_guess  #  xi
    y[3,:] = dy2_guess # dxi
    y[4,:] = y3_guess  #  a
    y[5,:] = dy3_guess # da  

    kappa = kappaarray[ii]
    msq = msqarray[ii]   
    msqp = msqparray[ii]
    #solve
    res = solve_bvp(dd,bc,r,y,tol = 0.001)
    if res.status != 0:
        print("Warning: res.status for kappa[%d] = %f is %d " % (ii,kappa, res.status))
    # plots 
    y_plot   = res.sol(r)[0]
    dy_plot  = res.sol(r)[1]
    y2_plot  = res.sol(r)[2]
    dy2_plot = res.sol(r)[3]
    y3_plot  = res.sol(r)[4]
    dy3_plot = res.sol(r)[5]

    np.savetxt('testpy.dat', np.vstack((r,y_plot,dy_plot,y2_plot,dy2_plot,y3_plot,dy3_plot)).T, delimiter=', ')

    y_guess =   y_plot
    y2_guess =  y2_plot
    y3_guess =  y3_plot
    dy_guess =  dy_plot
    dy2_guess = dy2_plot
    dy3_guess = dy3_plot
    
    yy[:,ii] =  y_plot[:]
    yy2[:,ii] = y2_plot[:]
    yy3[:,ii] = y3_plot[:]

    # Energy density
    tension = 0.0
    for i in range(N):
        e[i,ii] = 0.5*(dy_plot[i])**2 + 0.5*(y_plot[i])**2*(n + h*y3_plot[i])**2/(r[i])**2  \
        +0.5 * msq * (y_plot[i])**2 + 0.25*lmb * (y_plot[i])**4  \
        +0.5 * (dy2_plot[i])**2 + 0.5 * (y2_plot[i])**2*(npp + hp*y3_plot[i])**2/(r[i])**2 \
        +0.5*msqp*(y2_plot[i])**2 + 0.25*lmbp*(y2_plot[i])**4  \
        +0.5*kappa*(y_plot[i]*y2_plot[i])**2  \
        +0.5*(dy3_plot[i]/r[i])**2 \
        + 0.25*lmb * v**4 + 0.25*lmbp * vp**4 + 0.5 * kappa * v**2 * vp**2
        tension = tension + r[i]*e[i,ii]
  
    tension = 2*math.pi*(r[1]-r[0])*tension
    for i in range(N):
        print(r[i],y_plot[i],dy_plot[i],y2_plot[i],dy2_plot[i],y3_plot[i],dy3_plot[i],e[i,ii],ii,kappa, file = datafile)


    print(tension,'#Tension od the string = ',tension*(246/v)**2,'GeV^2','Mass of a string = ', 10**42 * tension*(246/v)**2 *10**(-27),file = datafile)
    print(' ',file = datafile)
    print(' ',file = datafile)
    #print(' ',file = datafile)

    # Consistency check
    print('#kappa = ',kappa,file = datafile2)
    for i in range(math.floor((N-1)/10)):
        xxx = dy_plot[i]/r[i] - (n+h*y3_plot[i])**2*y_plot[i]/(r[i]**2) - msq*y_plot[i]-lmb*y_plot[i]**3 -kappa*y_plot[i]*y2_plot[i]**2 + (dy_plot[i+1]-dy_plot[i])/r[i]
        print(r[i],xxx, file = datafile2)  

 
fig = plt.figure(figsize=(8,5))    
ax1 = fig.add_axes([0.10,0.10,0.70,0.85])

for ii in range(NN):
    ax1.plot(r,yy[:,ii],r,yy2[:,ii],r,yy3[:,ii],c=cmap(ii))
    #ax1.plot(r,yy[:,ii],c=cmap(ii))
    
norm = mpl.colors.Normalize(vmin = k0,vmax = kf)
sm   = plt.cm.ScalarMappable(cmap=cmap, norm= norm)
sm.set_array([])
cb = plt.colorbar(sm)
#cbarlabel = r'$k$'
cb.set_label('$\kappa$', labelpad=-45, y=1.05, rotation=0, fontsize = 16)
#plt.text(18,1.5,'$\\xi(r)$',fontsize=16)
#plt.text(18,0.1,'$\phi(r)$',fontsize = 16)
#plt.text(18,3.5,'$a(r)$',fontsize = 16)
plt.xlabel('$r$',fontsize =16)
#plt.ylabel('$\\varepsilon$',fontsize =16)
#plt.xlim([0,20])
plt.title("$n=$%d, $h=$%02.1f, $n'=$%d, $h'=$%02.1f, $\lambda =$%02.1f, $\lambda'=$%02.1f, $v =$%03.2f, $v'=$%02.1f"%(n,h,npp,hp,lmb,lmbp,v,vp))
plt.show()
  
 
