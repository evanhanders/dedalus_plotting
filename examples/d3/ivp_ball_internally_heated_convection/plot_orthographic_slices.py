"""
This script plots snapshots of the evolution of a 2D slice of an S2 surface using an orthographic projection.

Usage:
    plot_orthographic_slices.py [options]

Options:
    --root_dir=<str>         Path to root directory containing data_dir [default: .]
    --data_dir=<str>         Name of data handler directory [default: slices]
    --out_name=<str>         Name of figure output directory & base name of saved figures [default: snapshots_orthographic]
    --start_fig=<int         Number of first figure file [default: 1]
    --start_file=<int>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<int>          Total number of files to plot
    --dpi=<int>              Image pixel density [default: 200]

    --col_inch=<float>       Number of inches / column [default: 3]
    --row_inch=<float>       Number of inches / row [default: 3]
"""

from docopt import docopt

args = docopt(__doc__)
from plotpal.slices import SlicePlotter

root_dir = args["--root_dir"]
data_dir = args["--data_dir"]
start_fig = int(args["--start_fig"])
start_file = int(args["--start_file"])
out_name = args["--out_name"]
n_files = args["--n_files"]
if n_files is not None:
    n_files = int(n_files)

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

# remove_mean option removes the numpy mean of the data
plotter.setup_grid(num_cols=2, num_rows=1, orthographic=True, **plotter_kwargs)
plotter.add_orthographic_colormesh(
    "T(r=0.5)", azimuth_basis="phi", colatitude_basis="theta", remove_mean=True
)
plotter.add_orthographic_colormesh(
    "T(r=1)", azimuth_basis="phi", colatitude_basis="theta", remove_mean=True
)
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args["--dpi"]))
