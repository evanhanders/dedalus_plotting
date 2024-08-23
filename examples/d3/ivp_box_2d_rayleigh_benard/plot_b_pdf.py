"""
Script for plotting probability distribution functions of 2D quantities.

Usage:
    plot_pdfs.py [options]

Options:
    --root_dir=<str>     Path to root directory containing data_dir [default: .]
    --data_dir=<str>     Name of data handler directory [default: snapshots]
    --out_name=<str>     Name of figure output directory & base name of saved figures [default: b_pdf]
    --start_fig=<int>    Number of first figure file [default: 1]
    --start_file=<int>   Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<float>    Total number of files to plot
    --dpi=<int>          Image pixel density [default: 200]
    --bins=<int>         Number of bins per pdf [default: 50]

    --col_inch=<float>   Number of inches / column [default: 6]
    --row_inch=<float>   Number of inches / row [default: 5]
"""

import logging

logger = logging.getLogger(__name__)
from docopt import docopt

args = docopt(__doc__)
from plotpal.pdfs import PdfPlotter

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

bins = int(args["--bins"])

# Load in figures and make plots
threeD = False
bases = ["x", "z"]
pdfs_to_plot = [
    "b",
]
plotter = PdfPlotter(
    run_dir=root_dir,
    sub_dir=data_dir,
    out_name=out_name,
    start_file=start_file,
    num_files=n_files,
)
plotter.calculate_pdfs(
    pdfs_to_plot, bins=bins, threeD=threeD, bases=bases, uneven_basis="z"
)
plotter.plot_pdfs(
    dpi=int(args["--dpi"]),
    row_inch=float(args["--row_inch"]),
    col_inch=float(args["--col_inch"]),
)
