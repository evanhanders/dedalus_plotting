import logging
from collections import OrderedDict
from dataclasses import dataclass
from sys import stdout
from typing import Callable, Optional, Union

import h5py  # type: ignore
import matplotlib
import numpy as np
from mpi4py import MPI

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 9})

from dedalus.extras.flow_tools import GlobalArrayReducer  # type: ignore

from plotpal.file_reader import RolledDset, SingleTypeReader, match_basis
from plotpal.plot_grid import PlotGrid, RegularPlotGrid

logger = logging.getLogger(__name__.split(".")[-1])


def save_dim_scale(
    dim: h5py.AttributeManager,
    scale_group: h5py.Group,
    task_name: str,
    scale_name: str,
    scale_data: np.ndarray,
    dtype: type = np.float64,
) -> None:
    """Saves a dimension scale from a HDF5 task to a different HDF5 group"""
    full_scale_name = "{} - {}".format(task_name, scale_name)
    scale_dset = scale_group.create_dataset(
        name=full_scale_name, shape=scale_data.shape, dtype=dtype
    )
    scale_dset[:] = scale_data
    scale_dset.make_scale(scale_name)
    dim.attach_scale(scale_dset)


class AveragedProfilePlotter(SingleTypeReader):
    """
    A class for breaking up profiles into evenly spaced sequential chunks and
    averaging them to reduce noise. Includes functionality to save the averaged
    profiles to hdf5 files and plot them.
    """

    def __init__(
        self,
        run_dir: str,
        sub_dir: str,
        out_name: str,
        distribution: str = "even-chunk",
        num_files: Optional[int] = None,
        roll_writes: Optional[int] = None,
        start_file: int = 1,
        global_comm: MPI.Intracomm = MPI.COMM_WORLD,
        chunk_size: int = 100,
    ):
        """Initialize the AveragedProfilePlotter"""
        super().__init__(
            run_dir=run_dir,
            sub_dir=sub_dir,
            out_name=out_name,
            distribution=distribution,
            num_files=num_files,
            roll_writes=roll_writes,
            start_file=start_file,
            global_comm=global_comm,
            chunk_size=chunk_size,
        )
        self.writes_per_avg = chunk_size
        self.plots: list[tuple[str, list[str], str, PlotGrid]] = []
        self.tasks: list[str] = []
        self.averages: dict[str, np.ndarray] = OrderedDict()
        self.stored_averages: dict[str, list[tuple[np.ndarray, int, float, float]]] = (
            OrderedDict()
        )
        self.stored_bases: dict[str, tuple[str, np.ndarray]] = OrderedDict()

    def add_average_plot(
        self,
        x_basis: str,
        y_tasks: Union[list[str], str],
        name: str,
        fig_height: float = 3,
        fig_width: float = 3,
    ) -> None:
        """
        Specifies a profile to average and plot.

        Parameters
        ----------
        x_basis : The basis that the profiles can be plotted against.
        y_tasks : A list of the tasks to plot.
        name : The name of the plot file.
        fig_height : The height of the figure in inches.
        fig_width : The width of the figure in inches.
        """
        if isinstance(y_tasks, str):
            y_tasks = [
                y_tasks,
            ]
        self.plots.append(
            (
                x_basis,
                y_tasks,
                name,
                RegularPlotGrid(
                    num_rows=1, num_cols=1, col_inch=fig_width, row_inch=fig_height
                ),
            )
        )
        for task in y_tasks:
            if task not in self.tasks:
                self.tasks.append(task)
                self.stored_averages[task] = []

    def plot_average_profiles(self, dpi: int = 200, save_data: bool = False) -> None:
        """
        Plots the averaged profiles.

        Parameters
        ----------
        dpi : int
            The resolution of the saved plots.
        save_data : bool
            Whether to save the averaged profiles to hdf5 files.
        """
        local_count = 0
        start_time = np.inf
        while self.writes_remain():
            dsets, ni = self.get_dsets(self.tasks)
            for task in self.tasks:
                if local_count == 0:  # Reset the average
                    self.averages[task] = np.zeros_like(dsets[task][ni, :].squeeze())
                    start_time = dsets[task].dims[0]["sim_time"][ni]
                self.averages[task] += dsets[task][
                    ni, :
                ].squeeze()  # Add the current profile to the average
            local_count += 1

            # Check if we have enough profiles to average
            if local_count == self.writes_per_avg:
                write_number = int(
                    dsets[task].dims[0]["write_number"][ni] / self.writes_per_avg
                )
                if self.comm.rank == 0:
                    print(
                        "writing average profiles; plot number {}".format(write_number)
                    )
                    stdout.flush()
                end_time = dsets[task].dims[0]["sim_time"][ni]

                # Loop over the plots and plot the profiles
                for plot_info in self.plots:
                    x_basis, y_tasks, name, grid = plot_info
                    ax = grid.axes["ax_0-0"]
                    for task in y_tasks:
                        # Get the x-profile to plot this y-profile against.
                        if task in self.stored_bases:
                            x = self.stored_bases[task][1]
                        else:
                            x = match_basis(dsets[task], x_basis)
                            self.stored_bases[task] = (x_basis, x)
                        # Divide by the number of profiles to get the average
                        y = self.averages[task] / local_count
                        ax.plot(x, y, label=task)
                    ax.set_xlabel(x_basis)
                    ax.legend()
                    plt.suptitle("t = {:.2e}-{:.2e}".format(start_time, end_time))
                    grid.fig.savefig(
                        "{:s}/{:s}_{:03d}.png".format(self.out_dir, name, write_number),
                        dpi=dpi,
                        bbox_inches="tight",
                    )
                    ax.clear()
                # If we're going to save the data, we need to store it.
                if save_data:
                    for task in self.tasks:
                        y = self.averages[task] / local_count
                        self.stored_averages[task].append(
                            (y, write_number, start_time, end_time)
                        )
                local_count = 0
        if save_data:
            self.save_averaged_profiles()

    def save_averaged_profiles(self) -> None:
        """Saves the averaged profiles to hdf5 files."""
        reducer = GlobalArrayReducer(self.comm)
        save_data = OrderedDict()
        for task in self.tasks:
            num_writes = int(
                reducer.reduce_scalar(self.stored_averages[task][-1][2], MPI.MAX)
            )
            out_data = np.zeros(
                (num_writes,) + self.stored_averages[task][-1][0].shape,
                dtype=np.float64,
            )
            out_start_times = np.zeros(num_writes, dtype=np.float64)
            out_dts = np.zeros_like(out_start_times)
            # fill out_data
            for avg, wn, start, end in self.stored_averages[task]:
                out_data[wn - 1, :] = avg
                out_start_times[wn - 1] = start
                out_dts[wn - 1] = end - start
            # broadcast and gather data on root node
            if self.comm.rank == 0:
                reduced_data: Optional[np.ndarray] = np.zeros_like(out_data)
                reduced_start_times: Optional[np.ndarray] = np.zeros_like(
                    out_start_times
                )
                reduced_dts: Optional[np.ndarray] = np.zeros_like(out_dts)
            else:
                reduced_data = reduced_start_times = reduced_dts = None
            self.comm.Reduce(out_data, reduced_data, op=MPI.SUM, root=0)
            self.comm.Reduce(out_start_times, reduced_start_times, op=MPI.SUM, root=0)
            self.comm.Reduce(out_dts, reduced_dts, op=MPI.SUM, root=0)
            save_data[task] = (reduced_data, reduced_start_times, reduced_dts)

        if self.comm.rank == 0:
            # save to file
            with h5py.File("{:s}/averaged_profiles.h5".format(self.out_dir), "w") as f:
                scale_group = f.create_group("scales")
                task_group = f.create_group("tasks")
                for task in self.tasks:
                    this_data, this_start_times, this_dts = save_data[task]
                    assert (
                        isinstance(this_data, np.ndarray)
                        and isinstance(this_start_times, np.ndarray)
                        and isinstance(this_dts, np.ndarray)
                    ), "Data must be numpy arrays; communication error occurred."
                    dset = task_group.create_dataset(
                        name=task, shape=this_data.shape, dtype=np.float64
                    )
                    dset[:] = this_data
                    dset.dims[0].label = "t"
                    for arr, sn in zip(
                        [this_start_times, this_dts], ["sim_time", "avg_time"]
                    ):
                        save_dim_scale(dset.dims[0], scale_group, task, sn, arr)

                    basis_name, basis = self.stored_bases[task]
                    dset.dims[1].label = basis_name
                    save_dim_scale(dset.dims[1], scale_group, task, basis_name, basis)


@dataclass
class ProfileLine:
    basis: str
    task: Union[
        str,
        Callable[
            [matplotlib.axes.Axes, dict[str, Union[h5py.Dataset, RolledDset]], int],
            None,
        ],
    ]
    grid_num: int
    needed_tasks: Optional[list[str]]
    ylim: tuple[Optional[float], Optional[float]]
    mpl_kwargs: dict[str, str]


class RolledProfilePlotter(SingleTypeReader):
    """
    A class for stepping through each profile in an IVP's output and plotting line plots of rolled
    averages to reduce noise and see evolution. Includes functionality to save the averaged
    profiles to hdf5 files and plot them.
    """

    def __init__(
        self,
        run_dir: str,
        sub_dir: str,
        out_name: str,
        distribution: str = "even-write",
        num_files: Optional[int] = None,
        roll_writes: Optional[int] = 20,
        start_file: int = 1,
        global_comm: MPI.Intracomm = MPI.COMM_WORLD,
        chunk_size: int = 1000,
    ):
        """Initialize the plotter."""
        super().__init__(
            run_dir=run_dir,
            sub_dir=sub_dir,
            out_name=out_name,
            distribution=distribution,
            num_files=num_files,
            roll_writes=roll_writes,
            start_file=start_file,
            global_comm=global_comm,
            chunk_size=chunk_size,
        )
        self.lines: list[ProfileLine] = []
        self.tasks: list[str] = []
        self.color_ind = 0

    def setup_grid(
        self,
        num_rows: int = 1,
        num_cols: int = 1,
        col_inch: float = 3,
        row_inch: float = 3,
        pad_factor: float = 10,
    ) -> None:
        """Initialize the plot grid for the line plots"""
        self.grid: PlotGrid = RegularPlotGrid(
            num_rows=num_rows,
            num_cols=num_cols,
            cbar=False,
            polar=False,
            mollweide=False,
            orthographic=False,
            threeD=False,
            col_inch=col_inch,
            row_inch=row_inch,
            pad_factor=pad_factor,
        )

    def use_custom_grid(self, custom_grid: PlotGrid) -> None:
        """Allows user to pass in a custom PlotGrid object."""
        self.grid = custom_grid

    def _groom_grid(self) -> list[matplotlib.axes.Axes]:
        """Assign colormeshes to axes subplots in the plot grid"""
        axs = []
        for nr in range(self.grid.nrows):
            for nc in range(self.grid.ncols):
                k = "ax_{}-{}".format(nr, nc)
                if k in self.grid.axes.keys():
                    axs.append(self.grid.axes[k])
        return axs

    def add_line(
        self,
        basis: str,
        task: Union[
            str,
            Callable[
                [matplotlib.axes.Axes, dict[str, Union[h5py.Dataset, RolledDset]], int],
                None,
            ],
        ],
        grid_num: int,
        needed_tasks: Optional[list[str]] = None,
        ylim: tuple[Optional[float], Optional[float]] = (None, None),
        mpl_kwargs: dict = {},
    ) -> None:
        """
        Specifies a profile to plot rolled averages of.

        Parameters
        ----------
            basis : The name of the dedalus basis for the x-axis
            task :
                If str, must be the name of a profile
                If function, must accept as arguments, in order:
                     1. a matplotlib subplot axis (ax),
                     2. a dictionary of datasets (dsets),
                     3. an integer for indexing (ni)
            grid_num :  Panel index for the plot
            ylim (optional) :  Y-limits for the plot
            mpl_kwargs : Keyword arguments to pass to the matplotlib.pyplot.plot function
        """
        if "color" not in mpl_kwargs and "c" not in mpl_kwargs:
            mpl_kwargs["color"] = "C" + str(self.color_ind)
            self.color_ind += 1
        if type(task) != str and needed_tasks is None:
            raise ValueError("must specify necessary tasks for your function.")
        if "label" not in mpl_kwargs:
            mpl_kwargs["label"] = task
        line = ProfileLine(
            basis=basis,
            task=task,
            grid_num=grid_num,
            needed_tasks=needed_tasks,
            ylim=ylim,
            mpl_kwargs=mpl_kwargs,
        )
        self.lines.append(line)

    def plot_lines(
        self, start_fig: int = 1, dpi: int = 200, save_profiles: bool = False
    ) -> None:
        """
        Plot figures of the rolled dedalus profiles at each timestep.

        # Arguments
            start_fig : The number in the filename for the first write.
            dpi : The pixel density of the output image
            save_profiles : If True, write an output file of all of the saved profiles.
        """
        with self.my_sync:
            if self.idle:
                return

            axs = self._groom_grid()
            tasks = []
            for line in self.lines:
                this_task = line.task
                if type(this_task) == str:
                    # plot is a simple handler output profile.
                    if this_task not in tasks:
                        tasks.append(this_task)
                else:
                    # plot is a function.
                    needed_tasks = line.needed_tasks
                    assert (
                        needed_tasks is not None
                    ), "Must specify needed tasks for Callable tasks."
                    for this_task in needed_tasks:
                        if this_task not in tasks:
                            tasks.append(this_task)

            saved_times: list[float] = []
            saved_writes: list[int] = []
            saved_profiles: dict[str, list[np.ndarray]] = OrderedDict()
            saved_bases: dict[str, np.ndarray] = OrderedDict()
            for k in tasks:
                saved_profiles[k] = []
            while self.writes_remain():
                assert (
                    self.current_file_handle is not None
                ), "File handle unable to be created and is None."
                dsets, ni = self.get_dsets(tasks)
                time_data = self.current_file_handle["scales"]

                if save_profiles:
                    saved_times.append(time_data["sim_time"][ni])
                    saved_writes.append(time_data["write_number"][ni])
                    for k in tasks:
                        saved_profiles[k].append(dsets[k][ni])

                # Plot the profiles
                for line in self.lines:
                    task = line.task
                    ylim = line.ylim
                    basis = line.basis
                    ax = axs[line.grid_num]
                    if type(task) == str:
                        dset = dsets[task]
                        x = match_basis(dset, basis)
                        if basis not in saved_bases.keys():
                            saved_bases[basis] = np.copy(x)
                        ax.plot(x, dset[ni].squeeze(), **line.mpl_kwargs)  # type: ignore
                    else:
                        assert callable(
                            task
                        ), "task must be a string or a callable function."
                        task(ax, dsets, ni)
                    if ylim[0] is not None or ylim[1] is not None:
                        ax.set_ylim(*ylim)
                    ax.legend()
                    ax.set_xlabel(basis)

                plt.suptitle("t = {:.4e}".format(time_data["sim_time"][ni]))

                self.grid.fig.savefig(
                    "{:s}/{:s}_{:06d}.png".format(
                        self.out_dir,
                        self.out_name,
                        int(time_data["write_number"][ni] + start_fig - 1),
                    ),
                    dpi=dpi,
                    bbox_inches="tight",
                )
                for ax in axs:
                    ax.clear()

            if save_profiles:
                self._save_profiles(
                    saved_bases, saved_profiles, saved_times, saved_writes
                )

    def _save_profiles(
        self,
        bases: dict[str, np.ndarray],
        profiles: dict[str, list[np.ndarray]],
        times_list: list[float],
        writes_list: list[int],
    ) -> None:
        """
        Saves time-averaged profiles to a file.

        Warning: this function can take a long time to run if there are many writes.

        # Arguments
            bases : NumPy arrays of dedalus basis grid points
            profiles : Lists of time-averaged profiles
            times : Lists of tuples of rough times of averaging intervals
            writes : List of write numbers
        """
        times = np.array(times_list)
        writes = np.array(writes_list, dtype=np.int64)
        # get total number of writes
        n_writes = np.zeros(
            1,
        )
        min_write = np.zeros(1, dtype=np.int64)
        n_writes[0] = writes.size
        min_write[0] = writes.min()
        self.comm.Allreduce(MPI.IN_PLACE, n_writes, op=MPI.SUM)
        self.comm.Allreduce(MPI.IN_PLACE, min_write, op=MPI.MIN)
        glob_writes = np.arange(n_writes[0], dtype=np.int64) + min_write[0]
        glob_times = np.zeros_like(glob_writes)
        local_slice = np.zeros_like(glob_writes, dtype=bool)
        local_slice[(glob_writes >= writes.min()) * (glob_writes <= writes.max())] = (
            True
        )
        glob_times[local_slice] = times
        self.comm.Allreduce(MPI.IN_PLACE, glob_times, op=MPI.SUM)

        glob_profs: dict[str, np.ndarray] = OrderedDict()
        for k, p in profiles.items():
            prof = np.array(p)
            glob_profs[k] = np.zeros((glob_times.size, *tuple(prof.shape[1:])))
            glob_profs[k][local_slice, :] = prof
            self.comm.Allreduce(MPI.IN_PLACE, glob_profs[k], op=MPI.SUM)

        if self.comm.rank == 0:
            with h5py.File(
                "{:s}/post_{:s}.h5".format(self.out_dir, self.out_name), "w"
            ) as f:
                for k, base in bases.items():
                    f[k] = base  # type: ignore
                for k, p in glob_profs.items():  # type: ignore
                    f[k] = p.squeeze()  # type: ignore
                f["sim_time"] = glob_times
                f["sim_writes"] = glob_writes
