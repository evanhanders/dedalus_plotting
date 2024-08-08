"""
This script plots snapshots of the evolution of a 2D slice through the equator of a 
simulation that spans both a BallBasis and a ShellBasis.

Usage:
    plot_equatorial_slices.py [options]

Options:
    --root_dir=<str>         Path to root directory containing data_dir [default: .]
    --data_dir=<str>         Name of data handler directory [default: slices]
    --out_name=<str>         Name of figure output directory & base name of saved figures [default: snapshots_equatorial]
    --start_fig=<int         Number of first figure file [default: 1]
    --start_file=<int>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<int>          Total number of files to plot
    --dpi=<int>              Image pixel density [default: 200]
    --r_inner=<float>        Inner radius of shell [default: 1.0]
    --r_outer=<float>        Outer radius of shell [default: 1.5]

    --col_inch=<float>       Number of inches / column [default: 3]
    --row_inch=<float>       Number of inches / row [default: 3]
"""

from docopt import docopt

args = docopt(__doc__)
from plotpal.slices import SlicePlotter

# Read in master output directory
root_dir = args["--root_dir"]
data_dir = args["--data_dir"]

# Read in additional plot arguments
start_fig = int(args["--start_fig"])
start_file = int(args["--start_file"])
out_name = args["--out_name"]
n_files = args["--n_files"]
if n_files is not None:
    n_files = int(n_files)

r_inner = float(args["--r_inner"])
r_outer = float(args["--r_outer"])

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(
    root_dir,
    file_dir=data_dir,
    out_name=out_name,
    start_file=start_file,
    n_files=n_files,
)
plotter_kwargs = {
    "col_inch": int(args["--col_inch"]),
    "row_inch": int(args["--row_inch"]),
}

# Just plot a single plot (1x1 grid) of the field "T eq"
# remove_x_mean option removes the (numpy horizontal mean) over phi
# divide_x_std divides the radial mean(abs(T eq)) over the phi direction
plotter.setup_grid(num_rows=1, num_cols=1, polar=True, **plotter_kwargs)
plotter.add_ball_shell_polar_colormesh(
    ball="TB eq",
    shell="TS eq",
    azimuth_basis="phi",
    radial_basis="r",
    remove_x_mean=True,
    divide_x_std=True,
    r_inner=r_inner,
    r_outer=r_outer,
)
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args["--dpi"]))
