"""
This example file is based on the d3 internally-heated convection example script.
A stable layer was added, and a ball and shell basis were coupled together.
"""

import logging
import sys
import time

import dedalus.public as d3
import numpy as np

logger = logging.getLogger(__name__)

# TODO: automate hermitian conjugacy enforcement
# TODO: finalize evaluators to save last output

# Define smooth Heaviside functions
from scipy.special import erf


def one_to_zero(x, x0, width=0.1):
    return (1 - erf((x - x0) / width)) / 2


def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)


# Allow restarting via command line
restart = len(sys.argv) > 1 and sys.argv[1] == "--restart"

# Parameters
Nphi, Ntheta, NrB, NrS = 128, 64, 48, 16
Rayleigh = 1e6
Prandtl = 1
dealias = 3 / 2
stop_sim_time = 25 + 25 * restart
timestepper = d3.SBDF2
max_timestep = 0.05
dtype = np.float64
r_inner, r_outer = 1, 1.5
mesh = None

# Bases
coords = d3.SphericalCoordinates("phi", "theta", "r")
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basisB = d3.BallBasis(
    coords, shape=(Nphi, Ntheta, NrB), radius=r_inner, dealias=dealias, dtype=dtype
)
basisS = d3.ShellBasis(
    coords,
    shape=(Nphi, Ntheta, NrS),
    radii=(r_inner, r_outer),
    dealias=dealias,
    dtype=dtype,
)
S2_basisB = basisB.S2_basis()
S2_basisS_bot = basisS.S2_basis(radius=r_inner)
S2_basisS_top = basisS.S2_basis(radius=r_outer)
phiB, thetaB, rB = basisB.local_grids((1, 1, 1))
phiS, thetaS, rS = basisS.local_grids((1, 1, 1))

# Fields
uB = dist.VectorField(coords, name="uB", bases=basisB)
pB = dist.Field(name="pB", bases=basisB)
TB = dist.Field(name="TB", bases=basisB)
uS = dist.VectorField(coords, name="uS", bases=basisS)
pS = dist.Field(name="pS", bases=basisS)
TS = dist.Field(name="TS", bases=basisS)
tau_uB = dist.VectorField(coords, name="tau uB", bases=S2_basisB)
tau_TB = dist.Field(name="tau TB", bases=S2_basisB)
tau_uS_bot = dist.VectorField(coords, name="tau uS bot", bases=S2_basisS_bot)
tau_TS_bot = dist.Field(name="tau TS bot", bases=S2_basisS_bot)
tau_uS_top = dist.VectorField(coords, name="tau uS top", bases=S2_basisS_top)
tau_TS_top = dist.Field(name="tau TS top", bases=S2_basisS_top)

# Substitutions
r_vecB = dist.VectorField(coords, bases=basisB.radial_basis)
r_vecB["g"][2] = rB
r_vecS = dist.VectorField(coords, bases=basisS.radial_basis)
r_vecS["g"][2] = rS
T_source = 6

stiffness = 100
grad_T0B = dist.VectorField(coords, bases=basisB.radial_basis)
grad_T0B["g"][2] = stiffness * zero_to_one(rB, 0.8, width=0.1)
grad_T0S = dist.VectorField(coords, bases=basisS.radial_basis)
grad_T0S["g"][2] = stiffness


kappa = (Rayleigh * Prandtl) ** (-1 / 2)
nu = (Rayleigh / Prandtl) ** (-1 / 2)

div = d3.Divergence
lap = lambda A: d3.Laplacian(A, coords)
grad = lambda A: d3.Gradient(A, coords)
curl = d3.Curl
dot = d3.DotProduct
cross = d3.CrossProduct
dt = d3.TimeDerivative
rad = d3.RadialComponent
ang = d3.AngularComponent
trans = d3.TransposeComponents
grid = d3.Grid

lift_basisB = basisB.clone_with(k=0)  # Natural output
liftB = lambda A, n: d3.Lift(A, lift_basisB, n)
lift_basisS = basisS.clone_with(k=2)
liftS = lambda A, n: d3.Lift(A, lift_basisS, n)

strain_rateB = grad(uB) + trans(grad(uB))
strain_rateS = grad(uS) + trans(grad(uS))

u_match_bc = uB(r=r_inner) - uS(r=r_inner)
p_match_bc = pB(r=r_inner) - pS(r=r_inner)
stress_match_bc = ang(rad(strain_rateB(r=r_inner) - strain_rateS(r=r_inner)))

T_match_bc = TB(r=r_inner) - TS(r=r_inner)
grad_T_match_bc = rad(grad(TB)(r=r_inner) - grad(TS)(r=r_inner))

shear_stress = ang(rad(strain_rateS(r=r_outer)))


# Problem
problem = d3.IVP(
    [
        pB,
        uB,
        TB,
        pS,
        uS,
        TS,
        tau_uB,
        tau_uS_bot,
        tau_TB,
        tau_TS_bot,
        tau_uS_top,
        tau_TS_top,
    ],
    namespace=locals(),
)

# Ball Eqns
problem.add_equation("div(uB) = 0")
problem.add_equation(
    "dt(uB) - nu*lap(uB) + grad(pB) - r_vecB*TB + liftB(tau_uB,-1) = - cross(curl(uB),uB)"
)
problem.add_equation(
    "dt(TB) + dot(uB, grad_T0B) - kappa*lap(TB) + liftB(tau_TB,-1) = - dot(uB,grad(TB)) + kappa*T_source"
)

# Shell eqns
problem.add_equation("div(uS) = 0")
problem.add_equation(
    "dt(uS) - nu*lap(uS) + grad(pS) - r_vecS*TS + liftS(tau_uS_bot,-1) + liftS(tau_uS_top, -2) = - cross(curl(uS),uS)"
)
problem.add_equation(
    "dt(TS) + dot(uS, grad_T0S) - kappa*lap(TS) + liftS(tau_TS_bot,-1) + liftS(tau_TS_top, -2) = - dot(uS,grad(TS)) + kappa*T_source"
)

# Vel match BCs
problem.add_equation("u_match_bc = 0")
problem.add_equation("p_match_bc = 0")
problem.add_equation("stress_match_bc = 0")

# Temp match BCs
problem.add_equation("T_match_bc = 0")
problem.add_equation("grad_T_match_bc = 0")

# Surface BCs
problem.add_equation("shear_stress = 0")  # stress free
problem.add_equation(
    "rad(uS(r=r_outer)) = 0", condition="ntheta != 0"
)  # no penetration
problem.add_equation("pS(r=r_outer) = 0", condition="ntheta == 0")  # pressure gauge
problem.add_equation("TS(r=r_outer) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if not restart:
    seed = 42 + dist.comm_cart.rank
    rand = np.random.RandomState(seed=seed)
    TB["g"] = 1 - rB**2
    TB["g"] += 0.5 * rand.rand(*TB["g"].shape) * one_to_zero(rB, 0.8, width=0.1)
    file_handler_mode = "overwrite"
    initial_timestep = max_timestep
else:
    write, initial_timestep = solver.load_state("checkpoints/checkpoints_s10.h5")
    initial_timestep = 8e-3
    file_handler_mode = "append"

# Averaging operations
volume = (4 / 3) * np.pi * r_outer**3
volumeB = (4 / 3) * np.pi * r_inner**3
volumeS = volume - volumeB
az_avg = lambda A: d3.Average(A, coords.coords[0])
s2_avg = lambda A: d3.Average(A, coords.S2coordsys)
vol_avgB = lambda A: d3.Integrate(A / volumeB, coords)
vol_avgS = lambda A: d3.Integrate(A / volumeS, coords)

# Analysis
slices = solver.evaluator.add_file_handler(
    "slices", sim_dt=0.1, max_writes=10, mode=file_handler_mode
)
slices.add_task(TB(phi=0), name="TB mer right")
slices.add_task(TB(phi=np.pi), name="TB mer left")
slices.add_task(TB(theta=np.pi / 2), name="TB eq")
slices.add_task(TB(r=0.5), name="TB r=0.5")
slices.add_task(TB(r=0.8), name="TB r=0.8")
slices.add_task(TB(r=1), name="TB r=1")
slices.add_task(TS(phi=0), name="TS mer right")
slices.add_task(TS(phi=np.pi), name="TS mer left")
slices.add_task(TS(theta=np.pi / 2), name="TS eq")
slices.add_task(TS(r=1.25), name="TS r=1.25")
slices.add_task(TS(r=1.45), name="TS r=1.45")

profiles = solver.evaluator.add_file_handler(
    "profiles", sim_dt=0.1, max_writes=10, mode=file_handler_mode
)
profiles.add_task(s2_avg(TB), name="TB profile")
profiles.add_task(s2_avg(TS), name="TS profile")

scalars = solver.evaluator.add_file_handler(
    "scalars", sim_dt=0.1, max_writes=10, mode=file_handler_mode
)
scalars.add_task(vol_avgB(dot(uB, uB) / 2), name="KE")

checkpoints = solver.evaluator.add_file_handler(
    "checkpoints", sim_dt=1, max_writes=1, mode=file_handler_mode
)
checkpoints.add_tasks(solver.state)

# CFL
CFL = d3.CFL(
    solver, initial_timestep, cadence=1, safety=0.35, threshold=0.1, max_dt=max_timestep
)
CFL.add_velocity(uB)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(dot(uB, uB), name="u2")

# Main loop
hermitian_cadence = 100
try:
    logger.info("Starting loop")
    start_time = time.time()
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration - 1) % 10 == 0:
            max_u = np.sqrt(flow.max("u2"))
            logger.info(
                "Iteration=%i, Time=%e, dt=%e, max(u)=%e"
                % (solver.iteration, solver.sim_time, timestep, max_u)
            )
        # Impose hermitian symmetry on two consecutive timesteps because we are using a 2-stage timestepper
        if solver.iteration % hermitian_cadence in [0, 1]:
            for f in solver.state:
                f.require_grid_space()
except:
    logger.error("Exception raised, triggering end of main loop.")
    raise
finally:
    end_time = time.time()
    logger.info("Iterations: %i" % solver.iteration)
    logger.info("Sim end time: %f" % solver.sim_time)
    logger.info("Run time: %.2f sec" % (end_time - start_time))
    logger.info(
        "Run time: %f cpu-hr" % ((end_time - start_time) / 60 / 60 * dist.comm.size)
    )
