reset

set terminal epslatex color colortext standalone

lambda = 1.0
lambdap = 1.0
n = 0
np = 2
h = 0.0#
hp = 1.0#np*h/n
v = 0.5
vp = 1.0

set output sprintf("figures/l=%1.2f_lp=%1.2f_n=%d_np=%d_h=%1.2f_hp=%1.2f_v=%1.2f_vp=%1.2f.tex",lambda,lambdap,n,np,h,hp,v,vp)
#set output sprintf("figures/l=%d_lp=%d_n=%d_np=%d_h=%d_hp=%d_v=%1.2f_vp=%d.tex",lambda,lambdap,n,np,h,hp,v,vp)

set cbrange [-sqrt(lambda*lambdap):sqrt(lambda*lambdap)]

set xlabel "$r$"
set cblabel "$\\kappa$" rotate by 0
set palette rgb 33,13,10
set label "$\\phi(r)$" at 18,v offset 0,-0.7
set label "$\\xi(r)$" at 18,vp offset 0,-0.7
set label "$a(r)$" at 18,-np/hp offset 0,0.7
#set title sprintf("$\\lambda = %d$, $\\lambda' = %d$, $n = %d$, $n' = %d$, $h = %1.1f$, $h' = %1.1f$, $v = %1.2f$, $v' = %d$",lambda, lambdap, n, np, h, hp, v, vp) offset 0.5,0
set title sprintf("$\\lambda = %d$, $\\lambda' = %d$, $n = %d$, $n' = %d$, $h = %d$, $h' = %d$, $v = %1.1f$, $v' = %d$",lambda, lambdap, n, np, h, hp, v, vp) offset 0.5,0

unset key
plot [0:20][-2.1:1.2] for [i=0:*] sprintf('data/profile_l=%1.2f_lp=%1.2f_n=%d_np=%d_h=%1.2f_hp=%1.2f_v=%1.2f_vp=%1.2f.dat',lambda,lambdap,n,np,h,hp,v,vp) i i u 1:2:10 w l palette lw 2, for [i=0:*] '' i i u 1:4:10 w l palette lw 2, for [i=0:*] ''i i u 1:6:10 w l palette lw 2
#set ylabel "$\\epsilon$"
#plot  [0:30][-0.05:] for [i=0:*] sprintf('data/profile_l=%1.2f_lp=%1.2f_n=%d_np=%d_h=%1.2f_hp=%1.2f_v=%1.2f_vp=%1.2f.dat',lambda,lambdap,n,np,h,hp,v,vp) i i u 1:8:10 w l palette lw 2
unset output
