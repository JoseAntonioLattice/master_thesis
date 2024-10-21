reset

#3b4992 #ee0000 #008b45 #631879 #008280

set terminal pdf 

set style line 1 linecolor '#3b4992' linewidth 3
set style line 2 linecolor '#ee0000' linewidth 3 dt 2

set xlabel 'r' font',16'


set title 'v = 1, {/Symbol l} = 1, n = 1' font',16'
set xlabel 'r' font',16'

set label 1 'f(r)'             font',16' at 3,0.85
set label 2 '{/symbol e}(r)' font',16' at 2,0.1  
set label 3 'v' font',16' at 9.5,.95


set output 'global_str.pdf'
pl [][-.1:1.1]'global_profile.dat' u 1:2 notitle w l ls 1, '' u 1:3 notitle w l ls 2
unset output
