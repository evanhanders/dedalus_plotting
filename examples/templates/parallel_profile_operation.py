"""
This script distributes output files across all MPI processes and then loops over 
every write, letting the user post-process e.g., 1D profiles to get scalar values
which are then properly sorted and saved.

Be sure to fill in or note all of the TODO's.

To see how it works, run the (d3) rayleigh_benard.py example, then type:
    mpirun -n 4 python3 parallel_profile_operation.py

Usage:
    parallel_profile_operation.py [options]

Options:
    --root_dir=<str>         Path to root directory containing data_dir [default: .]
    --data_dir=<str>         Name of dedalus handler [default: profiles]
    --out_dir=<str>          Name of output directory [default: post_profiles]
    --start_fig=<int>        Number of first figure file [default: 1]
    --start_file=<int>       Number of first Dedalus file to read [default: 1]
    --n_files=<int>          Total number of files to plot
    --dpi=<int>              Image pixel density [default: 200]
"""

from collections import OrderedDict

import h5py
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from mpi4py import MPI

args = docopt(__doc__)

from plotpal.file_reader import SingleTypeReader, match_basis

# Read in master output directory
root_dir = args["--root_dir"]
data_dir = args[
    "--data_dir"
]  # TODO: change default in docstring above to apply to your own simulation.

# Read in additional plot arguments
start_fig = int(args["--start_fig"])
start_file = int(args["--start_file"])
out_dir = args[
    "--out_dir"
]  # TODO: figures will be saved in out_dir/ by default; see below.
n_files = args["--n_files"]
dpi = int(args["--dpi"])
if n_files is not None:
    n_files = int(n_files)

reader = SingleTypeReader(
    run_dir=root_dir,
    sub_dir=data_dir,
    out_name=out_dir,
    num_files=n_files,
    distribution="even-write",
)
output_tasks = ["enstrophy"]  # TODO: Update this with the names of your output tasks.

post = OrderedDict()
post["sim_time"] = []
post["write_number"] = []
post["bulk_enstrophy"] = []
post["bound_enstrophy"] = []
first = True
while reader.writes_remain():
    dsets, ni = reader.get_dsets(output_tasks)
    time_data = dsets[output_tasks[0]].dims[0]
    post["sim_time"].append(time_data["sim_time"][ni])
    post["write_number"].append(time_data["write_number"][ni])

    # TODO: do desired output analyses here
    enstrophy = dsets["enstrophy"][ni].ravel()
    if first:
        z = match_basis(dsets["enstrophy"], "z")
        Lz = z.max() - z.min()
        dz = np.gradient(z)
        bulk = (z > z.min() + Lz / 10) * (z < z.max() - Lz / 10)  # inner 80% of domain.
        bound = np.invert(bulk)
    post["bulk_enstrophy"].append(
        np.sum((enstrophy * dz)[bulk]) / np.sum(dz[bulk])
    )  # Mean enstrophy in the bulk
    post["bound_enstrophy"].append(
        np.sum((enstrophy * dz)[bound]) / np.sum(dz[bound])
    )  # Mean enstrophy at the boundary

post["write_number"] = np.array(post["write_number"], dtype=int)

# Get max and min simulation write number to get total global writes.
buffer = np.zeros(1, dtype=int)
if reader.idle:
    buffer[0] = 0
else:
    buffer[0] = post["write_number"].max()
reader.reader.global_comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MAX)
global_max_write = buffer[0]
if reader.idle:
    buffer[0] = int(1e8)
else:
    buffer[0] = int(post["write_number"].min())
reader.reader.global_comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MIN)
global_min_write = buffer[0]

# Communicate scalar data across tasks in proper order.
num_writes = int(global_max_write - global_min_write + 1)
global_arr = np.zeros(num_writes, dtype=np.float64)
for field in ["sim_time", "bulk_enstrophy", "bound_enstrophy"]:
    if not reader.idle:
        indx = post["write_number"] - global_min_write
        global_arr[indx] = post[field]
    reader.reader.global_comm.Allreduce(MPI.IN_PLACE, global_arr, op=MPI.SUM)
    post[field] = np.copy(global_arr)
    global_arr *= 0

if reader.comm.rank == 0:
    # Plot enstrophy traces
    fig = plt.figure()
    plt.plot(post["sim_time"], post["bulk_enstrophy"], label="bulk")
    plt.plot(post["sim_time"], post["bound_enstrophy"], label="bound")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("enstrophy")
    fig.savefig(
        "{:s}/enstrophy_v_time.png".format(reader.out_dir), dpi=dpi, bbox_inches="tight"
    )

    # Save output file
    with h5py.File("{:s}/rolled_output.h5".format(reader.out_dir), "w") as f:
        for field in ["sim_time", "bulk_enstrophy", "bound_enstrophy"]:
            f[field] = post[field]
