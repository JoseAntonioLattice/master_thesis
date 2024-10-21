program vortices
  implicit none
  real*8 :: init, fin,r,l
  integer, parameter :: n = 6, m = 10
  real*8, parameter :: pi = acos(-1d0), xmin = -10d0, xmax = 10d0
  integer :: i, j
  real*8, dimension(n) :: x, y, radius
  real*8, dimension(m) :: theta
  
  init = 0.7d0
  fin = 1.3d0

  radius = [(0d0 + (10d0-0d0)/float(n)*i, i = 1,n)]

  theta = [(0d0 + (2*pi-0d0)/float(m)*j, j = 1,m)]

  x = [(xmin + (xmax-xmin)/float(n)*i, i = 1,n)]
  y = x

  open(unit = 100, file = 'configurations.dat')

  do i = 1, n
    do j = 1, m
      call random_number(r)
      r = init + r*(fin - init)
      !theta = atan(y(j)/x(i))
      write(100,*) radius(i)*cos(theta(j)),radius(i)*sin(theta(j)), -0.8*sin(theta(j)), 0.8*cos(theta(j))
    end do
  end do
  close(100)

end program vortices


