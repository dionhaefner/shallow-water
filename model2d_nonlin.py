import numpy as np
import matplotlib.pyplot as plt

# parameter
g, H0, f  = 9.81, 100.0, 2e-4
Nx, Ny = 200, 100
Delta_x, Delta_y = 5e3, 5e3
Delta_t = 0.125 * min(Delta_x, Delta_y) / np.sqrt(g * H0)
Lx, Ly = Delta_x * Nx, Delta_y * Ny
beta = 2e-11

c , R, Lx = np.sqrt(g*H0), np.sqrt(g*H0)/f,  Delta_x*Nx
print("Lx = ",Lx, " c = ",c," R = ",R)

# allocate memory
u, v, h = np.zeros((Ny,Nx)), np.zeros((Ny,Nx)), np.zeros((Ny,Nx))
du, dv, dh = np.zeros((2,Ny,Nx)), np.zeros((2,Ny,Nx)), np.zeros((2,Ny,Nx))
fe, fn = np.zeros((Ny,Nx)), np.zeros((Ny,Nx))
q, hc, Ke = np.zeros((Ny,Nx)),np.zeros((Ny,Nx)), np.zeros((Ny,Nx))
x, y = np.arange(Nx) * Delta_x, np.arange(Ny) * Delta_y
Y, X = np.meshgrid(y, x, indexing='ij')
F = f + Y * beta

# initial conditions
u[:,:] = 10 * np.exp(-(Y - y[Ny//2])**2 / (Lx * 0.02)**2)
# approximate balance h_y = -(f/g)u
h[...] = np.cumsum(-Delta_y * u * F / g, axis=0)
h[:,:] += H0 + 0.2 * np.sin(X / Lx * 10 * np.pi) * np.cos(Y / Ly * 8 * np.pi)


# boundary values of h must not be used
h[0,:], h[-1,:], h[:,0], h[:,-1] = np.nan, np.nan, np.nan, np.nan

# Adam-Bashforth coefficients for time stepping
A, B = 1.5+0.1, -(0.5+0.1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

tau, taum1 = 0, 1 # pointers to time levels

for n in range(10000000): # time step equations
  hc[1:-1,1:-1] = h[1:-1,1:-1]
  hc[0,:] = hc[1,:] # extrapolate to boundary values
  hc[-1,:] = hc[-2,:]
  hc[:,0] = hc[:,-1]
  hc[:,-1] = hc[:,-2]

  fe[1:-1,1:-1] = 0.5*(hc[1:-1,1:-1] + hc[1:-1,2:])*u[1:-1,1:-1]
  fn[1:-1,1:-1] = 0.5*(hc[1:-1,1:-1] + hc[2:,1:-1])*v[1:-1,1:-1]

  dh[tau,1:-1,1:-1] = -((fe[1:-1,1:-1]-fe[1:-1,:-2])/Delta_x
                     +(fn[1:-1,1:-1]-fn[:-2,1:-1])/Delta_y)
  # planetary and relative vorticity
  q[1:-1,1:-1] = F[1:-1,1:-1]+ ((v[1:-1,2:]-v[1:-1,1:-1])/Delta_x
                              - (u[2:,1:-1]-u[1:-1,1:-1])/Delta_y)
  # potential vorticity
  q[1:-1,1:-1] = q[1:-1,1:-1]/(0.25*(hc[1:-1,1:-1]+hc[1:-1,2:]+hc[2:,1:-1]+hc[2:,2:]))

  du[tau,1:-1,1:-1] = - g*(h[1:-1,2:]-h[1:-1,1:-1])/Delta_x
  dv[tau,1:-1,1:-1] = - g*(h[2:,1:-1]-h[1:-1,1:-1])/Delta_y
  du[tau,1:-1,1:-1] += +0.5*(q[1:-1,1:-1]*0.5*(fn[1:-1,1:-1]+fn[1:-1,2:])
                          + q[:-2 ,1:-1]*0.5*(fn[:-2 ,1:-1]+fn[:-2 ,2:]))
  dv[tau,1:-1,1:-1] += -0.5*(q[1:-1,1:-1]*0.5*(fe[1:-1,1:-1]+fe[2:,1:-1])
                          + q[1:-1 ,:-2]*0.5*(fe[1:-1, :-2]+fe[2: ,:-2]))
  Ke[1:-1,1:-1] = 0.5*(0.5*(u[1:-1,1:-1]**2 + u[1:-1,:-2 ]**2)
                    + 0.5*(v[1:-1,1:-1]**2 + v[:-2 ,1:-1]**2))
  du[tau,1:-1,1:-1] += -(Ke[1:-1,2: ]-Ke[1:-1,1:-1])/Delta_x
  dv[tau,1:-1,1:-1] += -(Ke[2: ,1:-1]-Ke[1:-1,1:-1])/Delta_y

  if n==0: # first time step: forward in time
    u[1:-1,1:-1] += Delta_t*du[tau,1:-1,1:-1]
    v[1:-1,1:-1] += Delta_t*dv[tau,1:-1,1:-1]
    h[1:-1,1:-1] += Delta_t*dh[tau,1:-1,1:-1]
  else: # later time steps: Adam-Bashforth interpolation
    u[1:-1,1:-1] += Delta_t*(A*du[tau,1:-1,1:-1]+B*du[taum1,1:-1,1:-1])
    v[1:-1,1:-1] += Delta_t*(A*dv[tau,1:-1,1:-1]+B*dv[taum1,1:-1,1:-1])
    h[1:-1,1:-1] += Delta_t*(A*dh[tau,1:-1,1:-1]+B*dh[taum1,1:-1,1:-1])

  u[:,-2] = 0  # boundary condition on u
  v[-2,:] = 0  # boundary condition on v

  # exchange pointers to time levels
  tau, taum1 = taum1, tau

  if np.mod(n,25)==0: # plotting
    # print(np.nansum(h-H0))
    ax.clear()
    cs = ax.pcolormesh(x/1e3,y/1e3,h,cmap='viridis')
    ax.quiver(x[::2]/1e3,y[::2]/1e3,u[::2,::2],v[::2,::2])
    ax.set_xlabel('$x/[km]$')
    ax.set_ylabel('$y/[km]$')
    ax.set_title('$h/[m]$ at t= %5.2f days, R=%5.1f km, c=%5.1f m/s '%(n*Delta_t/86400,R/1e3,c))
    plt.pause(0.01)
