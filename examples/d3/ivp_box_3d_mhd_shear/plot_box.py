"""
This script plots 3d volume rendering of top and sides of box
Usage:
    plot_box.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: snapshots]
    --out_name=<out_name>               Name of figure output directory & base name of saved figures [default: boxes]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.volumes import BoxPlotter

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
out_name    = args['--out_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

# Create Plotter object, tell it which fields to plot
plotter = BoxPlotter(root_dir, file_dir=data_dir, out_name=out_name, start_file=start_file, n_files=n_files)


# Just plot a single plot (1x1 grid) of the field "b"
plotter.setup_grid(num_rows=1, num_cols=2)
plotter.add_box(left='Bx side',right='Bx front', top='Bx bottom', x_basis='x', y_basis='y',z_basis='z', vmin=-7, vmax=7, cmap_exclusion=0.05, cmap='viridis')
plotter.add_cutout_box(left='Bx side',right='Bx front', top='Bx bottom', left_mid='By side', right_mid='By front', top_mid='By bottom', x_basis='x', y_basis='y',z_basis='z', vmin=-7, vmax=7, cmap_exclusion=0.05, cmap='inferno')
plotter.plot_boxes(start_fig=start_fig, dpi=int(args['--dpi']))
