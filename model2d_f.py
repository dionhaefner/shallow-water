""" model2d_f.py

2D shallow water model with Coriolis force.
"""

import numpy as np
import matplotlib.pyplot as plt

# set parameters
n_x = 100
dx = 20e3

n_y = 101
dy = 20e3

gravity = 9.81
depth = 100.
coriolis_param = 2e-4

dt = 0.5 * min(dx, dy) / np.sqrt(gravity * depth)

phase_speed = np.sqrt(gravity * depth)
rossby_radius = np.sqrt(gravity * depth) / coriolis_param

# plot parameters
plot_every = 2
max_quivers = 21

# grid setup
x, y = (
    np.arange(n_x) * dx,
    np.arange(n_y) * dy
)
Y, X = np.meshgrid(y, x, indexing='ij')

# initial conditions
h0 = 1.0 * np.exp(
    - (X - x[n_x // 2]) ** 2 / rossby_radius ** 2
    - (Y - y[n_y - 2]) ** 2 / rossby_radius ** 2
)
u0 = 0
v0 = 0


def update_plot(t, h, u, v, ax):
    h_range = 0.5 * np.max(np.abs(h0))
    quiver_stride = (
        slice(1, -1, n_y // max_quivers),
        slice(1, -1, n_x // max_quivers)
    )

    ax.clear()
    ax.pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        h[1:-1, 1:-1],
        vmin=-h_range, vmax=h_range, cmap='viridis'
    )
    ax.quiver(
        x[quiver_stride[1]] / 1e3,
        y[quiver_stride[0]] / 1e3,
        u[quiver_stride],
        v[quiver_stride]
    )
    ax.set_xlabel('$x$ (km)')
    ax.set_ylabel('$y$ (km)')
    ax.set_title(
        '$h$ (m) at t=%5.2f days, R=%5.1f km, c=%5.1f m/s '
        % (t / 86400, rossby_radius / 1e3, phase_speed)
    )
    plt.pause(0.01)


def iterate_shallow_water():
    # allocate arrays
    u, v, h = np.empty((n_y, n_x)), np.empty((n_y, n_x)), np.empty((n_y, n_x))

    # initial conditions
    h[...] = h0
    u[...] = u0
    v[...] = v0

    # boundary values of h must not be used
    h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = np.nan

    # time step equations
    while True:
        # update u
        v_avg = 0.25 * (v[1:-1, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[:-2, 2:])
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (
            + coriolis_param * v_avg
            - gravity * (h[1:-1, 2:] - h[1:-1, 1:-1]) / dx
        )
        u[:, -2] = 0

        # update v
        u_avg = 0.25 * (u[1:-1, 1:-1] + u[1:-1, :-2] + u[2:, 1:-1] + u[2:, :-2])
        v[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (
            - coriolis_param * u_avg
            - gravity * (h[2:, 1:-1] - h[1:-1, 1:-1]) / dy
        )
        v[-2, :] = 0

        # update h
        h[1:-1, 1:-1] = h[1:-1, 1:-1] - dt * depth * (
            (u[1:-1, 1:-1] - u[1:-1, :-2]) / dx
            + (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dy
        )

        yield h, u, v


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)

    model = iterate_shallow_water()
    for iteration, (h, u, v) in enumerate(model):
        if iteration % plot_every == 0:
            t = iteration * dt
            update_plot(t, h, u, v, ax)

        # stop if user closes plot window
        if not plt.fignum_exists(fig.number):
            break
