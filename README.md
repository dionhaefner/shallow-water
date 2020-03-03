# Shallow water modelling in Python

This repository contains two shallow water implementations in Python. They are a product of the [Bornö summer school 2018](https://nbiocean.bitbucket.io/bornoe2018b/), lead by [Markus Jochum](https://www.nbi.ku.dk/english/staff/?pure=en/persons/437464) and [Carsten Eden](https://www.ifm.uni-hamburg.de/en/institute/staff/eden.html).

![Nonlinear model spin-up](preview.gif?raw=true)

## Features

### Simple (linear) implementation

- Mass conserving on (Cartesian) Arakawa C-grid
- Mixed-time discretization
- Coriolis force on an f-plane
- Conditionally stable for `Δt >= √2 / f`

### Fully nonlinear implementation

All features of the simple implementation, plus...

- Adams-Bashforth time stepping scheme
- Lateral friction
- Varying Coriolis parameter (β-plane)
- Fully nonlinear momentum and continuity equations
- Energy conserving scheme by Sadourny (1975)
- Rigid or periodic boundary conditions
