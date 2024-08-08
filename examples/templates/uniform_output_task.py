"""
This script distributes output files across all MPI processes and then loops over 
every write, letting the user interact with dedlaus data to plot output in whatever way they want.

Be sure to fill in or note all of the TODO's.

To see how it works, run the (d3) rayleigh_benard.py example, then type:
    mpirun -n 4 python3 uniform_output_task.py

Usage:
    uniform_output_task.py [options]

Options:
    --root_dir=<str>         Path to root directory containing data_dir [default: .]
    --data_dir=<str>         Name of dedalus handler [default: snapshots]
    --out_dir=<str>          Name of output directory [default: frames]
    --start_fig=<int>        Number of first figure file [default: 1]
    --start_file=<int>       Number of first Dedalus file to read [default: 1]
    --n_files=<int>          Total number of files to plot
    --dpi=<int>              Image pixel density [default: 200]
"""

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

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
    root_dir, data_dir, out_dir, n_files=n_files, distribution="even-write"
)
output_tasks = ["b"]  # TODO: Update this with the names of your output tasks.

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
while reader.writes_remain():
    dsets, ni = reader.get_dsets(output_tasks)
    time_data = dsets[output_tasks[0]].dims[0]
    sim_time = time_data["sim_time"][ni]
    write_num = time_data["write_number"][ni]

    # TODO: do desired output analyses here
    b_data = dsets["b"][ni]
    x = match_basis(dsets["b"], "x")
    z = match_basis(dsets["b"], "z")
    zz, xx = np.meshgrid(z, x)
    ax.pcolormesh(xx, zz, b_data, cmap="RdBu_r", vmin=0, vmax=1)
    plt.suptitle("t = {:.4e}".format(sim_time))
    fig.savefig(
        "{:s}/{:s}_{:06d}.png".format(out_dir, out_dir, int(write_num + start_fig + 1)),
        dpi=dpi,
        bbox_inches="tight",
    )
    ax.clear()
