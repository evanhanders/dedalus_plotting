# Make-your-own templates

This directory contains a couple of template files for making your own postprocessing tasks using plotpal.

# parallel_profile_operation.py

This file shows how to use plotpal to do postprocessing operations of 1D Dedalus profiles in a fully parallelized fashion. This example file shows a simple measurement of the bulk vs boundary-layer enstrophy in rayleigh benard convection. It can be run as:

```sh
mpirun -n 2 python3 parallel_profile_operation.py
```

making an image like this:

[Enstrophy traces](./example_figs/enstrophy_v_time.png)

# uniform_output_task.py

This file shows how to use plotpal to access dedalus data and then do some kind of simple task to it (like plotting up a colormap). This specific example just reads the buoyancy field and plots a simple pcolormesh. It can be run as:

```sh
mpirun -n 2 python3 uniform_output_task.py
```

making a very minimal image like this that you can then modify:

[A simple colormesh](./example_figs/frames_000479.png)

