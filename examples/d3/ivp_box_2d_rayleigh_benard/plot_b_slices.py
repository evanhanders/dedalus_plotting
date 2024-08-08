"""
This script plots snapshots of dynamics in a 2D slice.

Usage:
    plot_b_slices.py [options]

Options:
    --root_dir=<str>     Path to root directory containing data_dir [default: .]
    --data_dir=<str>     Name of data handler directory [default: snapshots]
    --out_name=<str>     Name of figure output directory & base name of saved figures [default: b_frames]
    --start_fig=<int>    Number of first figure file [default: 1]
    --start_file=<int>   Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<float>    Total number of files to plot
    --dpi=<int>          Image pixel density [default: 200]

    --col_inch=<float>   Number of inches / column [default: 6]
    --row_inch=<float>   Number of inches / row [default: 1.5]
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

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(
    root_dir,
    file_dir=data_dir,
    out_name=out_name,
    start_file=start_file,
    n_files=n_files,
)
plotter_kwargs = {
    "col_inch": float(args["--col_inch"]),
    "row_inch": float(args["--row_inch"]),
}

plotter.setup_grid(num_rows=1, num_cols=1, **plotter_kwargs)
plotter.add_colormesh(
    "b", x_basis="x", y_basis="z", remove_x_mean=True, divide_x_std=True
)
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args["--dpi"]))
