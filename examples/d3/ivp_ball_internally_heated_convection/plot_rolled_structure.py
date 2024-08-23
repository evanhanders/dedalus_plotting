"""
This script plots the average temperature structure of the simulation, averaged over a specified number of writes.

Usage:
    plot_rolled_structure.py [options]

Options:
    --root_dir=<str>           Path to root directory containing data_dir [default: .]
    --data_dir=<str>           Name of data handler directory [default: profiles]
    --subdir_name=<str>        Name of figure output directory & base name of saved figures [default: rolled_structure]
    --start_file=<int>         Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<int>            Total number of files to plot
    --roll_writes=<int>        Number of writes over which to take average
    --dpi=<int>                Image pixel density [default: 200]

    --col_inch=<float>         Subplot width (inches) [default: 6]
    --row_inch=<float>         Subplot height (inches) [default: 3]
"""

from docopt import docopt

args = docopt(__doc__)
from plotpal.file_reader import match_basis
from plotpal.profiles import RolledProfilePlotter

root_dir = args["--root_dir"]
data_dir = args["--data_dir"]
start_file = int(args["--start_file"])
subdir_name = args["--subdir_name"]
n_files = args["--n_files"]
if n_files is not None:
    n_files = int(n_files)

roll_writes = args["--roll_writes"]
if roll_writes is not None:
    roll_writes = int(roll_writes)


def tot_lum(ax, dictionary, index):
    r = match_basis(dictionary["T profile"], "r")
    ax.plot(
        r,
        dictionary["conv luminosity"][index].ravel()
        + dictionary["cond luminosity"][index].ravel(),
        c="black",
        label="total luminosity",
    )


# Create Plotter object, tell it which fields to plot
plotter = RolledProfilePlotter(
    run_dir=root_dir,
    sub_dir=data_dir,
    out_name=subdir_name,
    roll_writes=roll_writes,
    start_file=start_file,
    num_files=n_files,
)
plotter.setup_grid(
    num_rows=2,
    num_cols=1,
    col_inch=float(args["--col_inch"]),
    row_inch=float(args["--row_inch"]),
)
plotter.add_line("r", "T profile", grid_num=0)
plotter.add_line(
    "r", tot_lum, grid_num=1, needed_tasks=["conv luminosity", "cond luminosity"]
)
plotter.add_line("r", "conv luminosity", grid_num=1)
plotter.add_line("r", "cond luminosity", grid_num=1)
plotter.plot_lines()
