reset

#3b4992 #ee0000 #008b45 #631879 #008280

set terminal pdf enhanced font 'Helvetica'

set palette model HSV functions gray,1,1 # HSV color space


set style line 1 linewidth 0.5
set style line 2 linewidth 0.5 dt 2
set style line 3 linewidth 0.5 dt 3
set style line 4 linewidth 0.5 dt 4


set title "v = {/Symbol l} = 1/2, v' = 2, {/Symbol l}' = 1, n = h = n' = h' = 1" font',16'
set xlabel 'r' font',16'

set label 1 'f(r)' font',16' at 2, 0.6  
set label 2 '{/Symbol x}(r)' font',16' at 2, 1.8
set label 3 'a(r)'           font',16' at 2,-0.75
set label 4 '{/symbol e}(r)' font',16' at 3,0.15


#set arrow from first 9.5,-0.5 to 9.9,-0.95
set label 5 "-n/h = -n'/h'" font',16' at 7.5,-.88
set label 6 'v' font',16' at 9,.65
set label 7 "v'" font',16' at 9,1.85

set label 8 "{/Symbol k}'" font',16' at 10.5,3

set output 'localBL_str.pdf'

plot [0:10][-1.1:2.1] for [j = 0:10]'profile.dat' i j u 1:2 notitle w l ls 1 lc palette frac j/10.0 ,\
                      for [j = 0:10]'' i j u 1:3 notitle w l ls 2 lc palette frac j/10.0,\
                      for [j = 0:10]'' i j u 1:4 notitle w l ls 3 lc palette frac j/10.0,\
                      for [j = 0:10]'' i j u 1:5 notitle w l ls 4 lc palette frac j/10.0

unset output
