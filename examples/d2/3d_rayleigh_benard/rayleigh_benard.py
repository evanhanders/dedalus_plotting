"""
This is a lightly modified example script from the Dedalus v2_master
examples/ivp directory. Runs with Dedalus v2.

Dedalus script for 3D Rayleigh-Benard convection.

This script uses parity-bases in the x and y directions to mimick stress-free,
insulating sidewalls.  The equations are scaled in units of the thermal
diffusion time (Pe = 1).

This script should be ran in parallel, and would be most efficient using a
2D process mesh.  It uses the built-in analysis framework to save 2D data slices
in HDF5 files.  The `merge_procs` command can be used to merge distributed analysis
sets from parallel runs, and the `plot_slices.py` script can be used to plot
the slices.

To run and merge using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py

The simulation should take roughly 2 process-minutes to run, and will
automatically stop after an hour.

"""

import logging
import time

import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post

logger = logging.getLogger(__name__)


# Parameters
Lx, Ly, Lz = (5.0, 5.0, 1.0)
Ra = 1e4
Pr = 1.0

# Stop criterion
tb = 1 / np.sqrt(Ra * Pr)
stop_sim_time = 40 * tb  # Buoyancy units.
stop_wall_time = 60 * 60.0


# Create bases and domain
start_init_time = time.time()
x_basis = de.SinCos("x", 32, interval=(0, Lx), dealias=3 / 2)
y_basis = de.SinCos("y", 32, interval=(0, Ly), dealias=3 / 2)
z_basis = de.Chebyshev("z", 32, interval=(-Lz / 2, Lz / 2), dealias=3 / 2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(
    domain, variables=["p", "b", "u", "v", "w", "bz", "uz", "vz", "wz"], time="t"
)
problem.meta["p", "b", "w", "bz", "wz"]["x", "y"]["parity"] = 1
problem.meta["v", "vz"]["x"]["parity"] = 1
problem.meta["v", "vz"]["y"]["parity"] = -1
problem.meta["u", "uz"]["x"]["parity"] = -1
problem.meta["u", "uz"]["y"]["parity"] = 1
problem.parameters["P"] = 1
problem.parameters["R"] = Pr
problem.parameters["F"] = F = Ra * Pr
problem.parameters["Lz"] = Lz
problem.parameters["Ly"] = Ly
problem.parameters["Lx"] = Lx

problem.add_equation("dx(u) + dy(v) + wz = 0")
problem.add_equation(
    "dt(b) - P*(dx(dx(b)) + dy(dy(b)) + dz(bz))             = - u*dx(b) - v*dy(b) - w*bz"
)
problem.add_equation(
    "dt(u) - R*(dx(dx(u)) + dy(dy(u)) + dz(uz)) + dx(p)     = - u*dx(u) - v*dy(u) - w*uz"
)
problem.add_equation(
    "dt(v) - R*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + dy(p)     = - u*dx(v) - v*dy(v) - w*vz"
)
problem.add_equation(
    "dt(w) - R*(dx(dx(w)) + dy(dy(w)) + dz(wz)) + dz(p) - b = - u*dx(w) - v*dy(w) - w*wz"
)
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

problem.add_bc("left(b) = -left(F*z)")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(b) = -right(F*z)")
problem.add_bc("right(u) = 0")
problem.add_bc("right(v) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("integ_z(p) = 0", condition="(nx == 0) and (ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info("Solver built")

# Initial conditions
z = domain.grid(2)
b = solver.state["b"]
bz = solver.state["bz"]

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert = 1e-3 * noise * (zt - z) * (z - zb)
b["g"] = -F * (z - pert)
b.differentiate("z", out=bz)

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time
solver.stop_iteration = np.inf

# Analysis
snap = solver.evaluator.add_file_handler("snapshots", sim_dt=0.5 * tb, max_writes=50)
for field in ["b", "w"]:
    snap.add_task(
        "interp({}, z={})".format(field, 0.99 * Lz / 2),
        scales=4,
        name="{} top".format(field),
    )
    snap.add_task(
        "interp({}, x=0)".format(field), scales=4, name="{} yz side".format(field)
    )
    snap.add_task(
        "interp({}, y=0)".format(field), scales=4, name="{} xz side".format(field)
    )
    snap.add_task(
        "interp({}, z=0)".format(field), scales=4, name="{} xy midplane".format(field)
    )
    snap.add_task(
        "interp({}, x={})".format(field, 0.5 * Lx),
        scales=4,
        name="{} yz midplane".format(field),
    )
    snap.add_task(
        "interp({}, y={})".format(field, 0.5 * Ly),
        scales=4,
        name="{} xz midplane".format(field),
    )

profiles = solver.evaluator.add_file_handler(
    "profiles",
    sim_dt=0.1 * tb,
    max_writes=50,
)
profiles.add_task("integ(b, 'x', 'y')/Lx/Ly", name="b")
profiles.add_task("integ(w*b, 'x', 'y')/Lx/Ly", name="conv_flux")
profiles.add_task("integ(-P*(bz), 'x', 'y')/Lx/Ly", name="cond_flux")

scalars = solver.evaluator.add_file_handler("scalars", sim_dt=0.1 * tb, max_writes=1e6)
scalars.add_task("integ(sqrt(u*u + w*w)/R)/Lx/Ly/Lz", name="Re")
scalars.add_task("1 + integ(w*b) / integ(-P*(bz))", name="Nu")

file_handlers = [snap, profiles, scalars]

# CFL
timestep = 0.5 * tb
CFL = flow_tools.CFL(
    solver,
    initial_dt=timestep,
    cadence=5,
    safety=0.5,
    max_change=1.5,
    min_change=0.5,
    max_dt=timestep,
)
CFL.add_velocities(("u", "v", "w"))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + v*v + w*w) / R", name="Re")

# Main loop
end_init_time = time.time()
logger.info("Initialization time: %f" % (end_init_time - start_init_time))
try:
    logger.info("Starting loop")
    while solver.ok:
        timestep = CFL.compute_dt()
        solver.step(timestep)
        if (solver.iteration - 1) % 10 == 0:
            string = "Iteration: %i, Time: %e, dt: %e" % (
                solver.iteration,
                solver.sim_time,
                timestep,
            )
            string += ", Max Re = %f" % flow.max("Re")
            logger.info(string)
except:
    logger.error("Exception raised, triggering end of main loop.")
    raise
finally:
    solver.log_stats()
    for task in file_handlers:
        post.merge_analysis(task.base_path)
