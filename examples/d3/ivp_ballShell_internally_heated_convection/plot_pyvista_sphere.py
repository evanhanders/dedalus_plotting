"""
This script plots 3d volume rendering of a cutout sphere

Usage:
    plot_pyvista_sphere.py [options]

Options:
    --root_dir=<str>         Path to root directory containing data_dir [default: .]
    --data_dir=<str>         Name of data handler directory [default: slices]
    --out_name=<str>         Name of figure output directory & base name of saved figures [default: cutsphere_plots]
    --start_fig=<int         Number of first figure file [default: 1]
    --start_file=<int>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<int>          Total number of files to plot
    --size=<int>             Subplot pixel span [default: 500]
"""

from docopt import docopt

args = docopt(__doc__)
from plotpal.volumes import PyVistaSpherePlotter

root_dir = args["--root_dir"]
data_dir = args["--data_dir"]
out_name = args["--out_name"]
start_fig = int(args["--start_fig"])
start_file = int(args["--start_file"])
n_files = args["--n_files"]
if n_files is not None:
    n_files = int(n_files)

plotter = PyVistaSpherePlotter(
    root_dir,
    file_dir=data_dir,
    out_name=out_name,
    start_file=start_file,
    n_files=n_files,
)
plotter.setup_grid(num_rows=1, num_cols=2, size=int(args["--size"]))
eq_fields = ["TB eq", "TS eq"]
right_mer_fields = ["TB mer left", "TS mer left"]
left_mer_fields = ["TB mer right", "TS mer right"]
outer_shell = "TS r=1.45"
inner_shell = "TB r=0.8"
plotter.add_sphere(
    equator=eq_fields,
    left_meridian=left_mer_fields,
    right_meridian=right_mer_fields,
    outer_shell=outer_shell,
    view=0,
    cmap_exclusion=0.02,
    cmap="RdBu_r",
    label="T'",
    r_inner=0,
    max_r=1.45,
    remove_radial_mean=True,
    divide_radial_stdev=True,
)
plotter.add_sphere(
    equator=eq_fields,
    left_meridian=left_mer_fields,
    right_meridian=right_mer_fields,
    outer_shell=inner_shell,
    view=0,
    cmap_exclusion=0.02,
    cmap="RdBu_r",
    label="T' convection",
    r_inner=0,
    max_r=0.8,
    remove_radial_mean=True,
    divide_radial_stdev=True,
)
plotter.plot_spheres()
