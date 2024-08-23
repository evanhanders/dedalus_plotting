"""
This script plots snapshots of the evolution of a 2D slice through the meridion of a BallBasis simulation.

Usage:
    plot_meridional_slices.py [options]

Options:
    --root_dir=<str>         Path to root directory containing data_dir [default: .]
    --data_dir=<str>         Name of data handler directory [default: slices]
    --out_name=<str>         Name of figure output directory & base name of saved figures [default: snapshots_meridional]
    --start_fig=<int         Number of first figure file [default: 1]
    --start_file=<int>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<int>          Total number of files to plot
    --dpi=<int>              Image pixel density [default: 200]
    --radius=<float>         Outer radius of simulation domain [default: 1]

    --col_inch=<float>       Number of inches / column [default: 3]
    --row_inch=<float>       Number of inches / row [default: 3]
"""

from docopt import docopt

args = docopt(__doc__)
from plotpal.slices import SlicePlotter

# Read in master output directory
root_dir = args["--root_dir"]
data_dir = args["--data_dir"]
start_fig = int(args["--start_fig"])
start_file = int(args["--start_file"])
out_name = args["--out_name"]
n_files = args["--n_files"]
if n_files is not None:
    n_files = int(n_files)

radius = float(args["--radius"])

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(
    run_dir=root_dir,
    sub_dir=data_dir,
    out_name=out_name,
    start_file=start_file,
    num_files=n_files,
)
plotter_kwargs = {
    "col_inch": int(args["--col_inch"]),
    "row_inch": int(args["--row_inch"]),
    "pad_factor": 10,
}

# remove_x_mean option removes the (numpy horizontal mean) over phi
# divide_x_std divides the radial mean(abs(T mer left)) over the phi direction
plotter.setup_grid(num_cols=1, num_rows=1, polar=True, **plotter_kwargs)
plotter.add_meridional_colormesh(
    left_task="T(phi=pi)",
    right_task="T(phi=0)",
    colatitude_basis="theta",
    radial_basis="r",
    remove_x_mean=False,
    r_inner=0,
    r_outer=radius,
    label="T meridional",
)
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args["--dpi"]))
