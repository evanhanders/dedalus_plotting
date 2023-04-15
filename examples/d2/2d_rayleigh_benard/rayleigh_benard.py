"""
This is a lightly modified example copied from the examples/ivp directory
of Dedalus' v2_master branch. Runs using Dedalus v2.

Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5

This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.

To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py

The simulations should take a few process-minutes to run.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (4., 1.)
Prandtl = 1.
Rayleigh = 1e6

# Timestepping and output
dt = 0.125
stop_sim_time = 25
fh_mode = 'overwrite'

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', 64, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['F'] = F = 1
problem.parameters['Lx'] = Lx
problem.parameters['Lz'] = Lz
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz)) - F*w       = -(u*dx(b) + w*bz)")
problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     = -(u*dx(u) + w*uz)")
problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b = -(u*dx(w) + w*wz)")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("b(z='left') = 0")
problem.add_bc("u(z='left') = 0")
problem.add_bc("w(z='left') = 0")
problem.add_bc("b(z='right') = 0")
problem.add_bc("u(z='right') = 0")
problem.add_bc("w(z='right') = 0", condition="(nx != 0)")
problem.add_bc("integ(p) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions

# Initial conditions
x, z = domain.all_grids()
b = solver.state['b']
bz = solver.state['bz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-3 * noise * (zt - z) * (z - zb)
b['g'] = F * pert
b.differentiate('z', out=bz)

# Integration parameters
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=50, mode=fh_mode)
snapshots.add_task('p')
snapshots.add_task('b')
snapshots.add_task('u', name='ux')
snapshots.add_task('w', name='uz')

profiles = solver.evaluator.add_file_handler('profiles', sim_dt=0.1, max_writes=50, mode=fh_mode)
profiles.add_task("integ(b, 'x')/Lx", name='b')
profiles.add_task("integ(w*b, 'x')/Lx", name='conv_flux')
profiles.add_task("integ(-P*(bz - F), 'x')/Lx", name='cond_flux')

scalars = solver.evaluator.add_file_handler('scalars', sim_dt=0.1, max_writes=1e6, mode=fh_mode)
scalars.add_task("integ(sqrt(u*u + w*w)/R)/Lx/Lz", name='Re')
scalars.add_task("1 + integ(w*b) / integ(-P*(bz - F))", name='Nu')

file_handlers = [snapshots, profiles, scalars]

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + w*w) / R", name='Re')

# Main loop
try:
    logger.info('Starting loop')
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            string = 'Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt)
            string += ', Max Re = %f' %flow.max('Re')
            logger.info(string)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
    for task in file_handlers:
        post.merge_analysis(task.base_path)
