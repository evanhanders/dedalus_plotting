"""
This script plots snapshots of the evolution of a 2D slice through the equator of a BallBasis simulation.

Usage:
    plot_slices.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: snapshots]
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: slice_frames]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 4]
    --row_inch=<in>                     Number of inches / row [default: 2]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
fig_name    = args['--fig_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_inch' : float(args['--col_inch']), 'row_inch' : float(args['--row_inch']) }

# Just plot a single plot (1x1 grid) of the field "b"
# remove_x_mean option removes the (numpy horizontal mean) over phi
# divide_x_mean divides the radial mean(abs(b)) over the x-direction
plotter.setup_grid(num_rows=1, num_cols=1, **plotter_kwargs)
plotter.add_colormesh('b', x_basis='x', y_basis='z', remove_x_mean=True, divide_x_mean=True)
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))