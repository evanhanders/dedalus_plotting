"""
Script for plotting traces of evaluated scalar quantities vs. time.
If --roll_writes is specified, both an instantaneous and rolled trace are plotted.

Usage:
    plot_scalar_traces.py [options]

Options:
    --root_dir=<str>        Path to root directory containing data_dir [default: .]
    --data_dir=<str>        Name of data handler directory [default: scalars]
    --out_name=<str>        Output directory for figures [default: traces]
    --start_file=<int>      Dedalus output file to start at [default: 1]
    --n_files=<int>         Number of files to plot [default: 100000]
    --dpi=<int>             Image pixel density [default: 150]

    --roll_writes=<int>     Number of writes to roll a comparison average over
"""

from docopt import docopt

args = docopt(__doc__)
from plotpal.scalars import ScalarFigure, ScalarPlotter

root_dir = args["--root_dir"]
data_dir = args["--data_dir"]
out_name = args["--out_name"]
start_file = int(args["--start_file"])
n_files = int(args["--n_files"])
dpi = int(args["--dpi"])

roll_writes = args["--roll_writes"]
if roll_writes is not None:
    roll_writes = int(roll_writes)

figs = []

# Nu vs time
fig1 = ScalarFigure(num_rows=2, num_cols=1, col_inch=6, fig_name="fundamentals")
fig1.add_field(0, "Re", color="orange")
fig1.add_field(1, "Nu", color="indigo")
figs.append(fig1)

# Load in figures and make plots
plotter = ScalarPlotter(
    root_dir,
    file_dir=data_dir,
    out_name=out_name,
    start_file=start_file,
    n_files=n_files,
    roll_writes=roll_writes,
)
plotter.load_figures(figs)
plotter.plot_figures(dpi=dpi)
plotter.plot_convergence_figures(dpi=dpi)
