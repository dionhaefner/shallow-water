import numpy as np
import pylab as plt


def periodic_boundary_x(x):  # periodic boundary conditions
  x[:, 0], x[:, -1] = x[:, -2], x[:, 1]


# parameter
Nx, Ny = 100, 52
Delta_x, Delta_y, Delta_t = 10e3, 10e3,  50.
g, H0, f = 9.81, 100.0, 2e-4

fac = 2
Nx = Nx*fac
Delta_x = Delta_x/fac
Ny = Ny*fac
Delta_y = Delta_y/fac
Delta_t = Delta_t/fac

Ah = 1e-3*f*Delta_x**2  # Delta_x/100.


c, R, Lx, Ly = np.sqrt(g*H0), np.sqrt(g*H0)/f,  Delta_x*Nx, Delta_y*Ny
print("Lx = ", Lx, " c = ", c, " R = ", R, " Ah = ", Ah)

# allocate memory
u, v, h = np.zeros((Ny, Nx)), np.zeros((Ny, Nx)), np.zeros((Ny, Nx))
du, dv, dh = np.zeros((2, Ny, Nx)), np.zeros((2, Ny, Nx)), np.zeros((2, Ny, Nx))
fe, fn = np.zeros((Ny, Nx)), np.zeros((Ny, Nx))
q, hc, Ke = np.zeros((Ny, Nx)), np.zeros((Ny, Nx)), np.zeros((Ny, Nx))
x, y = np.array(range(Nx))*Delta_x, np.array(range(Ny))*Delta_y
Y, X = np.meshgrid(y, x, indexing='ij')
F = f+Y*2e-11

channel = True
linear_momentum_equation = False


# initial conditions
#h[:,:] = H0+ (  40.*np.exp( -(X-x[3*Nx/4])**2/(R/.5)**2
#                           - (Y-y[1*Nx/4])**2/(R/.5)**2 ) )
#h[:,:] = H0+ ( 40.*np.exp( -(X-x[Nx/2])**2/(R)**2 - (Y-y[Ny-1])**2/(R)**2 ) )

# jet like initial condition

u[:, :] = 10*np.exp(-((Y-y[Ny//2]))**2/(Lx*0.02)**2)
# approximate balance h_y = -(f/g)u
for k in range(1, Ny-1):
   h[k+1, :] = h[k, :]-Delta_y*u[k, :]*F[k, :]/g
h[:, :] += H0 + 0.2*np.sin(X/Lx*10*np.pi)*np.cos(Y/Ly*8*np.pi)

if channel:
  h[0, :], h[-1, :] = np.nan, np.nan
  u[0, :], u[-1, :] = 0, 0
  periodic_boundary_x(h)
  periodic_boundary_x(u)
else:
  # boundary values of h must not be used
  h[0, :], h[-1, :], h[:, 0], h[:, -1] = np.nan, np.nan, np.nan, np.nan

# Adam-Bashforth coefficients for time stepping
A, B = 1.5+0.1,  -(0.5+0.1)

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)

tau = 0
taum1 = 1  # pointers to time levels

for n in range(10000000):  # time step equations

  fe[1:-1, 1:-1] = 0.5*(h[1:-1, 1:-1] + h[1:-1, 2:])*u[1:-1, 1:-1]
  fn[1:-1, 1:-1] = 0.5*(h[1:-1, 1:-1] + h[2:, 1:-1])*v[1:-1, 1:-1]
  if channel:
    periodic_boundary_x(fe)
  else:
    fe[:, -2] = 0  # boundary condition on u
  fn[-2, :] = 0  # boundary condition on v
  dh[tau, 1:-1, 1:-1] = -((fe[1:-1, 1:-1]-fe[1:-1, :-2])/Delta_x
                          + (fn[1:-1, 1:-1]-fn[:-2, 1:-1])/Delta_y)

  if linear_momentum_equation:
    v_av = 0.25*(v[1:-1, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[:-2, 2:])
    du[tau, 1:-1, 1:-1] = +F[1:-1, 1:-1]*v_av-g*(h[1:-1, 2:]-h[1:-1, 1:-1])/Delta_x
    u_av = 0.25*(u[1:-1, 1:-1] + u[1:-1, :-2] + u[2:, 1:-1] + u[2:, :-2])
    dv[tau, 1:-1, 1:-1] = -F[1:-1, 1:-1]*u_av-g*(h[2:, 1:-1]-h[1:-1, 1:-1])/Delta_y

  else:  # nonlinear momentum equation
    hc[1:-1, 1:-1] = h[1:-1, 1:-1]
    hc[0, :] = hc[1, :]  # extrapolate to boundary values
    hc[-1, :] = hc[-2, :]
    if channel:
      periodic_boundary_x(hc)
    else:
      hc[:, 0] = hc[:, 1]
      hc[:, -1] = hc[:, -2]
    # planetary and relative vorticity
    q[1:-1, 1:-1] = F[1:-1, 1:-1] + ((v[1:-1, 2:]-v[1:-1, 1:-1])/Delta_x
                                     - (u[2:, 1:-1]-u[1:-1, 1:-1])/Delta_y)
    # potential vorticity
    q[1:-1, 1:-1] = q[1:-1, 1:-1]/(0.25*(hc[1:-1, 1:-1]+hc[1:-1, 2:]+hc[2:, 1:-1]+hc[2:, 2:]))
    if channel:
      periodic_boundary_x(q)
    du[tau, 1:-1, 1:-1] = - g*(h[1:-1, 2:]-h[1:-1, 1:-1])/Delta_x
    dv[tau, 1:-1, 1:-1] = - g*(h[2:, 1:-1]-h[1:-1, 1:-1])/Delta_y
    du[tau, 1:-1, 1:-1] += +0.5*(q[1:-1, 1:-1]*0.5*(fn[1:-1, 1:-1]+fn[1:-1, 2:])
                                 + q[:-2, 1:-1]*0.5*(fn[:-2, 1:-1]+fn[:-2, 2:]))
    dv[tau, 1:-1, 1:-1] += -0.5*(q[1:-1, 1:-1]*0.5*(fe[1:-1, 1:-1]+fe[2:, 1:-1])
                                 + q[1:-1, :-2]*0.5*(fe[1:-1, :-2]+fe[2:, :-2]))
    Ke[1:-1, 1:-1] = 0.5*(0.5*(u[1:-1, 1:-1]**2 + u[1:-1, :-2]**2)
                          + 0.5*(v[1:-1, 1:-1]**2 + v[:-2, 1:-1]**2))
    if channel:
      periodic_boundary_x(Ke)
    du[tau, 1:-1, 1:-1] += -(Ke[1:-1, 2:]-Ke[1:-1, 1:-1])/Delta_x
    dv[tau, 1:-1, 1:-1] += -(Ke[2:, 1:-1]-Ke[1:-1, 1:-1])/Delta_y

  if n == 0:  # first time step: forward in time
    u[1:-1, 1:-1] += Delta_t*du[tau, 1:-1, 1:-1]
    v[1:-1, 1:-1] += Delta_t*dv[tau, 1:-1, 1:-1]
    h[1:-1, 1:-1] += Delta_t*dh[tau, 1:-1, 1:-1]
  else:  # later time steps: Adam-Bashforth interpolation
    u[1:-1, 1:-1] += Delta_t*(A*du[tau, 1:-1, 1:-1]+B*du[taum1, 1:-1, 1:-1])
    v[1:-1, 1:-1] += Delta_t*(A*dv[tau, 1:-1, 1:-1]+B*dv[taum1, 1:-1, 1:-1])
    h[1:-1, 1:-1] += Delta_t*(A*dh[tau, 1:-1, 1:-1]+B*dh[taum1, 1:-1, 1:-1])

  v[-2, :] = 0  # boundary condition on v
  if channel:
    periodic_boundary_x(u)
    periodic_boundary_x(v)
    periodic_boundary_x(h)
  else:
    u[:, -2] = 0  # boundary condition on u

  # lateral friction
  fe[1:-1, 1:-1] = Ah*(u[1:-1, 2:]-u[1:-1, 1:-1])/Delta_x
  fn[1:-1, 1:-1] = Ah*(u[2:, 1:-1]-u[1:-1, 1:-1])/Delta_y
  if channel:
    periodic_boundary_x(fe)
  else:
    fe[:, -2] = 0  # boundary condition on u
  fn[-2, :] = 0  # boundary condition on v
  u[1:-1, 1:-1] += Delta_t*((fe[1:-1, 1:-1]-fe[1:-1, :-2])/Delta_x
                            + (fn[1:-1, 1:-1]-fn[:-2, 1:-1])/Delta_y)

  fe[1:-1, 1:-1] = Ah*(v[1:-1, 2:]-u[1:-1, 1:-1])/Delta_x
  fn[1:-1, 1:-1] = Ah*(v[2:, 1:-1]-u[1:-1, 1:-1])/Delta_y
  if channel:
    periodic_boundary_x(fe)
  else:
    fe[:, -2] = 0  # boundary condition on u
  fn[-2, :] = 0  # boundary condition on v
  v[1:-1, 1:-1] += Delta_t*((fe[1:-1, 1:-1]-fe[1:-1, :-2])/Delta_x
                            + (fn[1:-1, 1:-1]-fn[:-2, 1:-1])/Delta_y)

  # exchange pointers to time levels
  tau = np.mod(tau+1, 2)
  taum1 = np.mod(taum1+1, 2)

  if np.mod(n*Delta_t/3600, 1.) == 0:  # plotting
    ax.clear()
    ax.contourf(x/1e3, y/1e3, h, np.linspace(-8 + H0, 8 + H0, 30), cmap=plt.get_cmap('RdBu_r'))
    #ax.contour(x/1e3,y/1e3,h,colors='k')
    ax.quiver(x[::4]/1e3, y[::4]/1e3, u[::4, ::4], v[::4, ::4])
    ax.set_xlabel('$x/[km]$')
    ax.set_ylabel('$y/[km]$')
    ax.set_title('$u/[m/s]$, $h/[m]$ at t= %5.2f h, R=%5.1f km, c=%5.1f m/s ' %
                 (n*Delta_t/3600, R/1e3, c))
    plt.pause(0.01)
