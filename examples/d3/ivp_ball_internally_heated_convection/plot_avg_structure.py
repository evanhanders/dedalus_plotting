"""
This script plots the average temperature structure of the simulation, averaged over a specified number of writes.

Usage:
    plot_avg_structure.py [options]

Options:
    --root_dir=<str>           Path to root directory containing data_dir [default: .]
    --data_dir=<str>           Name of data handler directory [default: profiles]
    --subdir_name=<str>        Name of figure output directory & base name of saved figures [default: avg_structure]
    --start_file=<int>         Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<int>            Total number of files to plot
    --writes_per_avg=<int>     Number of writes over which to take average [default: 50]
    --dpi=<int>                Image pixel density [default: 200]

    --fig_width=<float>        Figure width (inches) [default: 6]
    --fig_height=<float>       Figure height (inches) [default: 3]
"""

from docopt import docopt

args = docopt(__doc__)
from plotpal.profiles import AveragedProfilePlotter

root_dir = args["--root_dir"]
data_dir = args["--data_dir"]
start_file = int(args["--start_file"])
writes_per_avg = int(args["--writes_per_avg"])
subdir_name = args["--subdir_name"]
n_files = args["--n_files"]
if n_files is not None:
    n_files = int(n_files)

# Create Plotter object, tell it which fields to plot
plotter = AveragedProfilePlotter(
    run_dir=root_dir,
    chunk_size=writes_per_avg,
    sub_dir=data_dir,
    out_name=subdir_name,
    start_file=start_file,
    num_files=n_files,
)
plotter_kwargs = {
    "fig_width": int(args["--fig_width"]),
    "fig_height": int(args["--fig_height"]),
}
plotter.add_average_plot(
    x_basis="r", y_tasks="T profile", name="T_vs_r", **plotter_kwargs
)
plotter.add_average_plot(
    x_basis="r",
    y_tasks=("conv luminosity", "cond luminosity"),
    name="lum_vs_r",
    **plotter_kwargs,
)

plotter.plot_average_profiles(dpi=int(args["--dpi"]), save_data=True)
