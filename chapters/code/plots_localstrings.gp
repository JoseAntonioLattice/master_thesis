reset

#3b4992 #ee0000 #008b45 #631879 #008280

set terminal pdf 

set style line 1 linecolor '#3b4992' linewidth 3
set style line 2 linecolor '#008b45' linewidth 3 dt 2
set style line 3 linecolor '#631879' linewidth 2 dt 3
set style line 4 linecolor '#ee0000' linewidth 3 dt 4


set title 'v = 1, {/Symbol l} = 1, n = 1, h = 1' font',16'
set xlabel 'r' font',16'

set label 1 'f(r)'             font',16' at 3,0.85  
set label 2 'a(r)'                       font',16' at 3,-0.73
set label 3 '{/symbol e}(r)' font',16' at 3,0.1  


#set arrow from first 9.5,-0.5 to 9.9,-0.95
set label 4 '-n/h' font',16' at 9,-.9
set label 5 'v' font',16' at 9,.9

set output 'local_str.pdf'
pl [][-1.1:1.1]'local_profile.dat' u 1:2 notitle w l ls 1,'' u 1:3 notitle w l ls 2,'' u 1:4 notitle w l ls 4#, [3:]1 ls 3 notitle, [4:]-1 ls 3 notitle,
unset output
