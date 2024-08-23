"""
This script plots 3d volume rendering of slices across the top and sides of a
Cartesian Dedalus simulation on a box.

Usage:
    plot_box.py [options]

Options:
    --root_dir=<str>     Path to root directory containing data_dir [default: .]
    --data_dir=<str>     Name of data handler directory [default: snapshots]
    --out_name=<str>     Name of figure output directory & base name of saved figures [default: pyvista_box]
    --start_fig=<int>    Number of first figure file [default: 1]
    --start_file=<int>   Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<float>    Total number of files to plot
    --size=<int>         Image pixels per panel [default: 500]

"""

from docopt import docopt

args = docopt(__doc__)
from plotpal.volumes import PyVistaBoxPlotter

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
size = int(args["--size"])

plotter = PyVistaBoxPlotter(
    run_dir=root_dir,
    sub_dir=data_dir,
    out_name=out_name,
    start_file=start_file,
    num_files=n_files,
)
plotter.setup_grid(num_rows=2, num_cols=2, size=size)
plotter.add_box(
    left="w yz side",
    right="w xz side",
    top="w top",
    x_basis="x",
    y_basis="y",
    z_basis="z",
    cmap_exclusion=0.05,
    cmap="PuOr_r",
    label="Vertical Velocity",
)
plotter.add_cutout_box(
    left="w yz side",
    right="w xz side",
    top="w top",
    left_mid="w yz midplane",
    right_mid="w xz midplane",
    top_mid="w xy midplane",
    x_basis="x",
    y_basis="y",
    z_basis="z",
    cmap_exclusion=0.05,
    stretch=0.01,
    cmap="PuOr_r",
    label="Vertical Velocity",
)
plotter.add_box(
    left="b yz side",
    right="b xz side",
    top="b top",
    x_basis="x",
    y_basis="y",
    z_basis="z",
    cmap_exclusion=0.05,
    cmap="RdBu_r",
    label="Buoyancy Fluc",
    remove_x_mean=True,
)
plotter.add_cutout_box(
    left="b yz side",
    right="b xz side",
    top="b top",
    left_mid="b yz midplane",
    right_mid="b xz midplane",
    top_mid="b xy midplane",
    x_basis="x",
    y_basis="y",
    z_basis="z",
    cmap_exclusion=0.05,
    stretch=0.01,
    cmap="RdBu_r",
    label="Buoyancy Fluc",
    remove_x_mean=True,
)
plotter.plot_boxes()
