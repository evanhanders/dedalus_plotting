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
    --dpi=<int>              Image pixel density [default: 200]
    --radius=<float>         Outer radius of simulation domain [default: 1]
"""

from docopt import docopt

args = docopt(__doc__)
from plotpal.volumes import PyVistaSpherePlotter

root_dir = args["--root_dir"]
data_dir = args["--data_dir"]
out_name = args["--out_name"]
start_fig = int(args["--start_fig"])
start_file = int(args["--start_file"])
radius = float(args["--radius"])
n_files = args["--n_files"]
if n_files is not None:
    n_files = int(n_files)

plotter = PyVistaSpherePlotter(
    run_dir=root_dir,
    sub_dir=data_dir,
    out_name=out_name,
    start_file=start_file,
    num_files=n_files,
)
plotter.setup_grid(num_rows=1, num_cols=1, size=500)
plotter.add_sphere(
    equator=[
        "T(equator)",
    ],
    left_meridian=[
        "T(phi=0)",
    ],
    right_meridian=[
        "T(phi=pi)",
    ],
    outer_shell="T(r=1)",
    view=0,
    cmap_exclusion=0.02,
    cmap="RdBu_r",
    label="T'",
    r_inner=0,
    max_r=radius,
    remove_radial_mean=True,
    divide_radial_stdev=True,
)
plotter.plot_spheres()
