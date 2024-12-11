reset

set terminal epslatex color colortext standalone

lambda = 1.0
lambdap = 1.0
n = 0
np = 2
h = 0
hp = 1.0 #np*h/n
v = 0.5
vp = 1.0

xi = 0.0
xf = 5.0
yi = 0.85
yf = 1.15

xri = 0
xrf = 50
yri = -1.1
yrf = 1.2

set output sprintf("figures/zoom_l=%1.2f_lp=%1.2f_n=%d_np=%d_h=%1.2f_hp=%1.2f_v=%1.2f_vp=%1.2f.tex",lambda,lambdap,n,np,h,hp,v,vp)

set cbrange [-sqrt(lambda*lambdap):sqrt(lambda*lambdap)]

set xlabel "$r$"
set cblabel "$\\kappa$" rotate by 0
set palette rgb 33,13,10
set label "$\\phi(r)$" at 45,v offset 0,0.7
set label "$\\xi(r)$" at 45,vp offset 0,0.7
set label "$a(r)$" at 45,-n/h offset 0,0.7

set xrange[xri:xrf]
set yrange[yri:yrf]
unset key

set style arrow 5 head size 1,2

#set title sprintf("$\\lambda = %d$, $\\lambda' = %d$, $n = %d$, $n' = %d$, $h = %1.1f$, $h' = %1.1f$, $v = %1.2f$, $v' = %d$",lambda, lambdap, n, np, h, hp, v, vp) offset 0.5,0
set title sprintf("$\\lambda = %d$, $\\lambda' = %d$, $n = %d$, $n' = %d$, $h = %d$, $h' = %d$, $v = %1.1f$, $v' = %d$",lambda, lambdap, n, np, h, hp, v, vp) offset 0.5,0
     

set multiplot 
    
    set object 1 rect from xi,yi to xf,yf front fs empty border lc black 
    set arrow from xf,yi to 20,0.3 front 
    #set object 1 rect fc rgb 'white' fillstyle solid 0.0 noborder
    plot for [i=0:*] sprintf('data/profile_l=%1.2f_lp=%1.2f_n=%d_np=%d_h=%1.2f_hp=%1.2f_v=%1.2f_vp=%1.2f.dat',lambda,lambdap,n,np,h,hp,v,vp) i i u 1:2:10 w l palette lw 2, for [i=0:*] '' i i u 1:4:10 w l palette lw 2, for [i=0:*] ''i i u 1:6:10 w l palette lw 2
    
    set origin 0.3,0.22
    set size sq 0.4
    set xrange [xi:xf]
    set yrange [yi:yf]
    unset xlabel
    unset ylabel
    unset label
    
    unset title
    unset arrow
    set format y '\tiny %g'
    set format x '\tiny %g'
    unset colorbox
    p for [i=0:*] sprintf('data/profile_l=%1.2f_lp=%1.2f_n=%d_np=%d_h=%1.2f_hp=%1.2f_v=%1.2f_vp=%1.2f.dat',lambda,lambdap,n,np,h,hp,v,vp) i i u 1:4:10 w l palette lw 2 
unset multiplot
unset output

