from dataclasses import dataclass
from typing import Optional, Union, Any

import matplotlib.collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs #type: ignore
from cartopy.mpl.geoaxes import GeoAxes #type: ignore
matplotlib.rcParams.update({'font.size': 9})
import h5py #type: ignore
from mpi4py import MPI

from plotpal.file_reader import SingleTypeReader, match_basis, RolledDset
from plotpal.plot_grid import RegularColorbarPlotGrid, PlotGrid

import logging
logger = logging.getLogger(__name__.split('.')[-1])


@dataclass
class Colormesh:
    """ A struct containing information about a slice colormesh plot    """

    # Positional Arguments
    task: str  # The profile task name

    #Keyword arguments
    vector_ind: Optional[int] = None # If not None, plot the vector component with this index. For use with d3 vector fields.
    x_basis: str = 'x' # The dedalus basis name that the profile spans in the horizontal-direction of the plot
    y_basis: str = 'z' # The dedalus basis name that the profile spans in the vertical-direction of the plot
    cmap: str = 'RdBu_r'  # The matplotlib colormap name used to display the data
    label: Optional[str] = None # A text label for the colorbar
    remove_mean: bool = False # If True, remove the mean value of the profile at each time
    remove_x_mean: bool = False # If True, remove the mean value over the axis plotted in the x- direction
    divide_x_std: bool = False # If True, divide by the stdev over the x- direction
    pos_def: bool = False # If True, field is positive definite and colormap should span from zero to max.
    vmin: Optional[float] = None # The minimum value of the colormap
    vmax: Optional[float] = None # The maximum value of the colormap
    log: bool = False # If True, plot the log of the field
    cmap_exclusion: float = 0.005 # The fraction of the colormap to exclude from the min/max values
    linked_cbar_cm: Optional['Colormesh'] = None # A Colormesh object that this object shares a colorbar with
    linked_profile_cm: Optional['Colormesh'] = None # A Colormesh object that this object shares a mean profile with
    transpose: bool = False # If True, transpose the colormap when plotting; useful when x_basis has an index after y_basis in dedalus data.
    first: bool = True # If True, this is the first time the plot is being made
    xx: Optional[np.ndarray] = None # The x-coordinates for use in the pcolormesh call
    yy: Optional[np.ndarray] = None # The y-coordinates for use in the pcolormesh call
    color_plot: Optional[matplotlib.collections.QuadMesh] = None # The pcolormesh object for the plot


    def _modify_field(self, field: np.ndarray) -> np.ndarray:
        """ Modify the colormap field before plotting; e.g., remove mean, etc. """

        self.removed_mean: int = 0
        self.divided_std: Union[np.ndarray, int] = 1
        if self.linked_profile_cm is not None:
            # Use the same mean and std as another Colormesh object if specified.
            self.removed_mean = self.linked_profile_cm.removed_mean
            self.divided_std = self.linked_profile_cm.divided_std
        else:

            #Remove specified mean
            if self.remove_mean:
                self.removed_mean = np.mean(field)
            elif self.remove_x_mean:
                self.removed_mean = np.mean(field, axis=0)

            #Scale field by the stdev to bring out low-amplitude dynamics.
            if self.divide_x_std:
                self.divided_std = np.std(field, axis=0)
                assert isinstance(self.divided_std, np.ndarray), "divide_x_std only works for 2D fields"
                if type(self) == MeridionalColormesh or type(self) == PolarColormesh:
                    if self.r_pad[0] == 0:
                        #set interior 4% of points to have a smoothly varying std
                        N = self.divided_std.shape[0] // 10
                        mean_val = np.mean(self.divided_std[:N])
                        bound_val = self.divided_std[N]
                        indx = np.arange(N)
                        smoother = mean_val + (bound_val - mean_val)*indx/N
                        self.divided_std[:N] = smoother
        field -= self.removed_mean
        field /= self.divided_std

        if self.log: 
            field = np.log10(np.abs(field))

        return field

    def _get_minmax(self, field: np.ndarray) -> tuple[float, float]:
        """ Get the min and max values of the specified field for the colormap """
        if self.linked_cbar_cm is not None:
            # Use the same min/max as another Colormesh object if specified.
            return self.linked_cbar_cm.current_vmin, self.linked_cbar_cm.current_vmax
        else:
            vals = np.sort(field.flatten())
            if self.pos_def:
                #If the profile is positive definite, set the colormap to span from the max/min to zero.
                vals = np.sort(vals)
                if np.mean(vals) < 0:
                    vmin, vmax = vals[int(self.cmap_exclusion*len(vals))], 0.
                else:
                    vmin, vmax = 0., vals[int((1-self.cmap_exclusion)*len(vals))]
            else:
                #Otherwise, set the colormap to span from the +/- abs(max) values.
                vals = np.sort(np.abs(vals))
                vmax = vals[int((1-self.cmap_exclusion)*len(vals))]
                vmin = -vmax

            if self.vmin is not None:
                vmin = self.vmin
            if self.vmax is not None:
                vmax = self.vmax

            return vmin, vmax

    def _get_pcolormesh_coordinates(self, dset: Union[h5py.Dataset, RolledDset] ) -> None:
        """ make the x and y coordinates for pcolormesh """
        x = match_basis(dset, self.x_basis)
        y = match_basis(dset, self.y_basis)
        self.yy, self.xx = np.meshgrid(y, x)

    def _setup_colorbar(
            self, 
            plot: matplotlib.collections.QuadMesh, 
            cax: matplotlib.axes.Axes, 
            vmin: float, 
            vmax: float
            ) -> matplotlib.colorbar.Colorbar:
        """ Create the colorbar on the axis 'cax' and label it """
        cb = plt.colorbar(plot, cax=cax, orientation='horizontal')
        assert cb.solids is not None, "cb.solids must be a matplotlib.collections.QuadMesh"
        cb.solids.set_rasterized(True)
        cb.set_ticks(())
        cax.text(-0.01, 0.5, r'$_{{{:.2e}}}^{{{:.2e}}}$'.format(vmin, vmax), transform=cax.transAxes, ha='right', va='center')
        if  self.linked_cbar_cm is None:
            if self.label is None:
                if self.vector_ind is not None:
                    cax.text(1.05, 0.5, '{:s}[{}]'.format(self.task, self.vector_ind), transform=cax.transAxes, va='center', ha='left')
                else:
                    cax.text(1.05, 0.5, '{:s}'.format(self.task), transform=cax.transAxes, va='center', ha='left')
            else:
                cax.text(1.05, 0.5, '{:s}'.format(self.label), transform=cax.transAxes, va='center', ha='left')
        return cb

    def plot_colormesh(
            self, 
            ax: matplotlib.axes.Axes, 
            cax: matplotlib.axes.Axes, 
            dset: Union[h5py.Dataset, RolledDset],
            ni: int, 
            mpl_kwargs: dict[str, Any] = {}
            ) -> tuple[matplotlib.collections.QuadMesh, matplotlib.colorbar.Colorbar]:
        """ 
        Plot the colormesh
        
        Parameters
        ----------
        ax : The axis to plot the colormesh on.
        cax : The axis to plot the colorbar on.
        dset : The dataset to plot.
        ni : The index of the time step to plot.
        mpl_kwargs : Additional keyword arguments to pass to matplotlib.pyplot.pcolormesh.
        """
        if self.first:
            self._get_pcolormesh_coordinates(dset)

        field = np.squeeze(dset[ni,:])
        vector_ind = self.vector_ind
        if vector_ind is not None:
            field = field[vector_ind,:]

        field = self._modify_field(field)
        vmin, vmax = self._get_minmax(field)
        self.current_vmin, self.current_vmax = vmin, vmax

        if 'rasterized' not in mpl_kwargs.keys():
            mpl_kwargs['rasterized'] = True
        if 'shading' not in mpl_kwargs.keys():
            mpl_kwargs['shading'] = 'nearest'

        if self.transpose:
            field = field.T
        assert self.xx is not None and self.yy is not None, "xx and yy must be set before plotting"
        self.color_plot = ax.pcolormesh(self.xx, self.yy, field.real, cmap=self.cmap, vmin=vmin, vmax=vmax, **mpl_kwargs)
        cb = self._setup_colorbar(self.color_plot, cax, vmin, vmax)
        self.first = False
        return self.color_plot, cb


class CartesianColormesh(Colormesh):
     """ Colormesh logic specific to Cartesian coordinates """

     def plot_colormesh(
            self, 
            ax: matplotlib.axes.Axes, 
            cax: matplotlib.axes.Axes, 
            dset: Union[h5py.Dataset, RolledDset],
            ni: int, 
            mpl_kwargs: dict[str, Any] = {}
            ) -> tuple[matplotlib.collections.QuadMesh, matplotlib.colorbar.Colorbar]:
        plot, cb = super().plot_colormesh(ax, cax, dset, ni, **mpl_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        return plot, cb


@dataclass
class PolarColormesh(Colormesh):
    """Colormesh logic specific to polar coordinates or equatorial slices in spherical coordinates"""

    radial_basis: str = 'r'
    azimuth_basis: str = 'phi'
    r_inner: float = 0
    r_outer: float = 1

    def __post_init__(self):
        self.x_basis = self.azimuth_basis
        self.y_basis = self.radial_basis
        self.r_pad = (self.r_inner, self.r_outer)

    def _modify_field(self, field: np.ndarray) -> np.ndarray:
        field = super()._modify_field(field)
        field = np.pad(field, ((0, 1), (1, 1)), mode='edge')
        field[-1,:] = field[0,:] #set 2pi value == 0 value.
        return field

    def _get_pcolormesh_coordinates(self, dset: Union[h5py.Dataset, RolledDset]) -> None:
        x = phi = match_basis(dset, self.azimuth_basis)
        y = r   = match_basis(dset, self.radial_basis)
        phi = np.append(x, 2*np.pi)
        r = np.pad(r, ((1,1)), mode='constant', constant_values=self.r_pad)
        self.yy, self.xx = np.meshgrid(r, phi)

    def plot_colormesh(
            self, 
            ax: matplotlib.axes.Axes, 
            cax: matplotlib.axes.Axes, 
            dset: Union[h5py.Dataset, RolledDset],
            ni: int, 
            mpl_kwargs: dict[str, Any] = {}
            ) -> tuple[matplotlib.collections.QuadMesh, matplotlib.colorbar.Colorbar]:
        plot, cb = super().plot_colormesh(ax, cax, dset, ni, **mpl_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, self.r_pad[1])
        ax.set_aspect(1)
        return plot, cb


class MollweideColormesh(Colormesh):
    """ Colormesh logic specific to Mollweide projections of S2 coordinates """

    colatitude_basis: str = 'theta'
    azimuth_basis: str = 'phi'

    def __post_init__(self):
        self.x_basis = self.azimuth_basis
        self.y_basis = self.colatitude_basis

    def _get_pcolormesh_coordinates(self, dset: Union[h5py.Dataset, RolledDset]) -> None:
        x = phi = match_basis(dset, self.azimuth_basis)
        y = theta = match_basis(dset, self.colatitude_basis)
        phi -= np.pi
        theta = np.pi/2 - theta
        self.yy, self.xx = np.meshgrid(theta, phi)

    def plot_colormesh(
            self,
            ax: matplotlib.axes.Axes, 
            cax: matplotlib.axes.Axes, 
            dset: Union[h5py.Dataset, RolledDset],
            ni: int, 
            mpl_kwargs: dict[str, Any] = {}
            ) -> tuple[matplotlib.collections.QuadMesh, matplotlib.colorbar.Colorbar]:
        plot, cb = super().plot_colormesh(ax, cax, dset, ni, **mpl_kwargs)
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        return plot, cb


class OrthographicColormesh(Colormesh):
    """ Colormesh logic specific to Orthographic projections of S2 coordinates """

   
    colatitude_basis: str = 'theta'
    azimuth_basis: str = 'phi'

    def __post_init__(self):
        self.x_basis = self.azimuth_basis
        self.y_basis = self.colatitude_basis
        try:
            self.transform = ccrs.PlateCarree()
        except:
            raise ImportError("Cartopy must be installed for plotpal Orthographic plots")

    def _get_pcolormesh_coordinates(self, dset: Union[h5py.Dataset, RolledDset]) -> None:
        phi = match_basis(dset, self.azimuth_basis)
        theta = match_basis(dset, self.colatitude_basis)
        phi *= 180/np.pi
        theta *= 180/np.pi
        phi -= 180
        theta -= 90
        self.yy, self.xx = np.meshgrid(theta, phi)

    def plot_colormesh(
            self,
            ax: GeoAxes, 
            cax: matplotlib.axes.Axes, 
            dset: Union[h5py.Dataset, RolledDset],
            ni: int, 
            mpl_kwargs: dict[str, Any] = {}
            ) -> tuple[matplotlib.collections.QuadMesh, matplotlib.colorbar.Colorbar]:
        plot, cb = super().plot_colormesh(ax, cax, dset, ni, **mpl_kwargs)
        ax.gridlines()
        return plot, cb


class MeridionalColormesh(Colormesh):
    """ Colormesh logic specific to meridional slices in spherical coordinates """

    radial_basis: str = 'r'
    colatitude_basis: str = 'theta'
    r_inner: float = 0
    r_outer: float = 1
    left : bool = False

    def __post_init__(self):
        self.x_basis = self.colatitude_basis
        self.y_basis = self.radial_basis
        self.r_pad = (self.r_inner, self.r_outer)

    def _modify_field(self, field: np.ndarray) -> np.ndarray:
        field = super()._modify_field(field)
        field = np.pad(field, ((1, 1), (1, 1)), mode='edge')
        return field

    def _get_pcolormesh_coordinates(self, dset: Union[h5py.Dataset, RolledDset]) -> None:
        theta = match_basis(dset, self.colatitude_basis)
        r     = match_basis(dset, self.radial_basis)
        theta = np.pad(theta, ((1,1)), mode='constant', constant_values=(np.pi,0))
        if self.left:
            theta = np.pi/2 + theta
        else:
            #right side
            theta = np.pi/2 - theta
        r = np.pad(r, ((1,1)), mode='constant', constant_values=self.r_pad)
        self.yy, self.xx = np.meshgrid(r, theta)

    def plot_colormesh(
            self,
            ax: matplotlib.axes.Axes, 
            cax: matplotlib.axes.Axes, 
            dset: Union[h5py.Dataset, RolledDset],
            ni: int, 
            mpl_kwargs: dict[str, Any] = {}
            ) -> tuple[matplotlib.collections.QuadMesh, matplotlib.colorbar.Colorbar]:
        plot, cb = super().plot_colormesh(ax, cax, dset, ni, **mpl_kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, self.r_pad[1])
        ax.set_aspect(1)
        return plot, cb


class SlicePlotter(SingleTypeReader):
    """
    A class for plotting colormeshes of 2D slices of dedalus data.
    """

    def __init__(
            self, 
            run_dir: str, 
            sub_dir: str, 
            out_name: str, 
            distribution: str = 'even-write',
            num_files: Optional[int] = None, 
            roll_writes: Optional[int] = None,
            start_file: int = 1,
            global_comm: MPI.Intracomm = MPI.COMM_WORLD,
            chunk_size: int = 1000
            ):
        """
        Initializes the slice plotter.
        """
        self.grid: Optional[PlotGrid] = None
        super(SlicePlotter, self).__init__(
            run_dir,
            distribution=distribution,
            sub_dir=sub_dir,
            out_name=out_name,
            num_files=num_files,
            start_file=start_file,
            global_comm=global_comm,
            chunk_size=chunk_size,
            roll_writes=roll_writes
            )
        self.counter = 0
        self.colormeshes: list[tuple[int, Colormesh]] = []

    def setup_grid(
            self, 
            num_rows: int = 1, 
            num_cols: int = 1, 
            cbar: bool = False, 
            polar: bool = False, 
            mollweide: bool = False, 
            orthographic: bool = False, 
            threeD: bool = False,
            col_inch: float = 3, 
            row_inch: float = 3, 
            pad_factor: float = 10
            ) -> None:
        """ Initialize the plot grid for the colormeshes """
        self.grid = RegularColorbarPlotGrid(
            num_rows=num_rows,
            num_cols=num_cols,
            cbar=cbar,
            polar=polar,
            mollweide=mollweide,
            orthographic=orthographic,
            threeD=threeD,
            col_inch=col_inch,
            row_inch=row_inch,
            pad_factor=pad_factor
        )

    def use_custom_grid(self, custom_grid: PlotGrid) -> None:
        """ Allows the user to use a custom grid """
        self.grid = custom_grid

    def add_colormesh(self, *args, **kwargs) -> None:
        self.colormeshes.append((self.counter, Colormesh(*args, **kwargs))) #type: ignore
        self.counter += 1

    def add_cartesian_colormesh(self, *args, **kwargs) -> None:
        self.colormeshes.append((self.counter, CartesianColormesh(*args, **kwargs))) #type: ignore
        self.counter += 1

    def add_polar_colormesh(self, *args, **kwargs) -> None: 
        self.colormeshes.append((self.counter, PolarColormesh(*args, **kwargs))) #type: ignore
        self.counter += 1

    def add_mollweide_colormesh(self, *args, **kwargs) -> None:
        self.colormeshes.append((self.counter, MollweideColormesh(*args, **kwargs))) #type: ignore
        self.counter += 1

    def add_orthographic_colormesh(self, *args, **kwargs) -> None:
        self.colormeshes.append((self.counter, OrthographicColormesh(*args, **kwargs))) #type: ignore
        self.counter += 1

    def add_meridional_colormesh(self, left_task: str, right_task: str, **kwargs) -> None:
        """ Adds a colormesh for a meridional slice of a spherical field.
        Must specify both left and right sides of the meridional slice. """
        #TODO: be smarter about setting up the colorbars -- do it based on both sides rather than just left.
        self.colormeshes.append((self.counter, MeridionalColormesh(left_task, left=True, **kwargs))) #type: ignore
        self.colormeshes.append((self.counter, MeridionalColormesh(right_task, linked_cbar_cm=self.colormeshes[-1][1], linked_profile_cm=self.colormeshes[-1][1], **kwargs))) #type: ignore
        self.counter += 1

    def add_ball_shell_polar_colormesh(
            self, 
            ball_task: str, 
            shell_task: str, 
            r_inner: Optional[float] = None, 
            r_outer: Optional[float] = None, 
            **kwargs
            ) -> None:
        """ Adds a colormesh for a polar / equatorial slice of a spherical field that spans a ball and a shell. """
        #TODO: fix how colorbar is set; currently set by the ball.
        self.colormeshes.append((self.counter, PolarColormesh(ball_task, r_inner=0, r_outer=r_inner, **kwargs))) #type: ignore
        self.colormeshes.append((self.counter, PolarColormesh(shell_task, r_inner=r_inner, r_outer=r_outer, linked_cbar_cm=self.colormeshes[-1][1], **kwargs))) #type: ignore
        self.counter += 1

    def add_ball_shell_meridional_colormesh(
            self, 
            ball_left_task: str, 
            ball_right_task: str,
            shell_left_task: str, 
            shell_right_task: str, 
            r_stitch: Optional[float] = None, 
            r_outer: Optional[float] = None, 
            **kwargs
            ) -> None:
        """ Adds a colormesh for a meridional slice of a spherical field that spans a ball and a shell.
            Must specify both left and right sides of the meridional slice for both the ball and the shell."""
        #TODO: fix how colorbar is set; currently set by the left side of the ball.
        self.colormeshes.append((self.counter, MeridionalColormesh(ball_left_task, left=True, r_inner=0, r_outer=r_stitch, **kwargs))) #type: ignore
        first_cm = self.colormeshes[-1][1]
        self.colormeshes.append((self.counter, MeridionalColormesh(ball_right_task, r_inner=0, r_outer=r_stitch, linked_cbar_cm=first_cm, linked_profile_cm=first_cm, **kwargs))) #type: ignore
        self.colormeshes.append((self.counter, MeridionalColormesh(shell_left_task, left=True, r_inner=r_stitch, r_outer=r_outer, linked_cbar_cm=first_cm, **kwargs))) #type: ignore
        self.colormeshes.append((self.counter, MeridionalColormesh(shell_right_task, r_inner=r_stitch, r_outer=r_outer, linked_cbar_cm=first_cm, linked_profile_cm=first_cm, **kwargs))) #type: ignore
        self.counter += 1

    def add_shell_shell_meridional_colormesh(
            self, 
            left_inner_shell: str, 
            left_outer_shell: str,
            right_inner_shell: str, 
            right_outer_shell: str,
            r_inner: float, 
            r_stitch: float, 
            r_outer: float, 
            **kwargs
            ) -> None:
        """ Adds a colormesh for a meridional slice of a spherical field that spans two shells. 
            Must specify both left and right sides of the meridional slice for both shells."""
        self.colormeshes.append((self.counter, MeridionalColormesh(left_inner_shell, left=True, r_inner=r_inner, r_outer=r_stitch, **kwargs))) #type: ignore
        first_cm = self.colormeshes[-1][1]
        self.colormeshes.append((self.counter, MeridionalColormesh(right_inner_shell, r_inner=r_inner, r_outer=r_stitch, linked_cbar_cm=first_cm, linked_profile_cm=first_cm, **kwargs))) #type: ignore
        self.colormeshes.append((self.counter, MeridionalColormesh(left_outer_shell, left=True, r_inner=r_stitch, r_outer=r_outer, linked_cbar_cm=first_cm, **kwargs))) #type: ignore
        outer_cm = self.colormeshes[-1][1]
        self.colormeshes.append((self.counter, MeridionalColormesh(right_outer_shell, r_inner=r_stitch, r_outer=r_outer, linked_cbar_cm=first_cm, linked_profile_cm=outer_cm, **kwargs))) #type: ignore
        self.counter += 1

    def add_ball_2shells_polar_colormesh(
            self, 
            fields: tuple[str,str,str], 
            r_stitches: tuple[float,float], 
            r_outer: float, 
            **kwargs
            ) -> None:
        """ Adds a colormesh for a polar / equatorial slice of a spherical field that spans a ball and two shells."""
        self.colormeshes.append((self.counter, PolarColormesh(fields[0], r_inner=0, r_outer=r_stitches[0], **kwargs))) #type: ignore
        self.colormeshes.append((self.counter, PolarColormesh(fields[1], r_inner=r_stitches[0], r_outer=r_stitches[1], linked_cbar_cm=self.colormeshes[-1][1], **kwargs))) #type: ignore
        self.colormeshes.append((self.counter, PolarColormesh(fields[2], r_inner=r_stitches[1], r_outer=r_outer, linked_cbar_cm=self.colormeshes[-2][1], **kwargs)))    #type: ignore
        self.counter += 1

    def add_ball_2shells_meridional_colormesh(
            self, 
            left_fields: tuple[str,str,str], 
            right_fields: tuple[str,str,str],
            r_stitches: tuple[float,float], 
            r_outer: float, 
            **kwargs
            ) -> None:
        """ Adds a colormesh for a meridional slice of a spherical field that spans a ball and two shells.
            Must specify both left and right sides of the meridional slice for the ball and both shells."""
        self.colormeshes.append((self.counter, MeridionalColormesh(left_fields[0], left=True, r_inner=0, r_outer=r_stitches[0], **kwargs))) #type: ignore
        first_cm = self.colormeshes[-1][1]
        self.colormeshes.append((self.counter, MeridionalColormesh(right_fields[0], left=False, r_inner=0, r_outer=r_stitches[0], linked_profile_cm=self.colormeshes[-1][1], linked_cbar_cm=first_cm, **kwargs))) #type: ignore
        self.colormeshes.append((self.counter, MeridionalColormesh(left_fields[1], left=True, r_inner=r_stitches[0], r_outer=r_stitches[1], linked_cbar_cm=first_cm, **kwargs))) #type: ignore
        self.colormeshes.append((self.counter, MeridionalColormesh(right_fields[1], left=False, r_inner=r_stitches[0], r_outer=r_stitches[1], linked_profile_cm=self.colormeshes[-1][1], linked_cbar_cm=first_cm, **kwargs))) #type: ignore
        self.colormeshes.append((self.counter, MeridionalColormesh(left_fields[2], left=True, r_inner=r_stitches[1], r_outer=r_outer, linked_cbar_cm=first_cm, **kwargs))) #type: ignore
        self.colormeshes.append((self.counter, MeridionalColormesh(right_fields[2], left=False, r_inner=r_stitches[1], r_outer=r_outer, linked_profile_cm=self.colormeshes[-1][1], linked_cbar_cm=first_cm, **kwargs))) #type: ignore
        self.counter += 1

    def _groom_grid(self) -> tuple[list[matplotlib.axes.Axes], list[matplotlib.axes.Axes]]:
        """ Assign colormeshes to axes subplots in the plot grid """
        assert self.grid is not None, "Must set up a grid before plotting"
        axs, caxs = [], []
        for nr in range(self.grid.nrows):
            for nc in range(self.grid.ncols):
                k = 'ax_{}-{}'.format(nr, nc)
                if k in self.grid.axes.keys():
                    axs.append(self.grid.axes[k])
                    caxs.append(self.grid.cbar_axes[k])
        return axs, caxs

    def plot_colormeshes(self, start_fig: int = 1, dpi: int = 200, **mpl_kwargs):
        """
        Plot figures of the 2D dedalus data slices at each timestep.

        # Arguments
            start_fig (int) :
                The number in the filename for the first write.
            dpi (int) :
                The pixel density of the output image
            kwargs :
                extra keyword args for matplotlib.pyplot.pcolormesh
        """
        with self.my_sync:
            assert self.grid is not None, "Must set up a grid before plotting"
            axs, caxs = self._groom_grid()
            tasks = []
            for k, cm in self.colormeshes:
                if cm.task not in tasks:
                    tasks.append(cm.task)
            if self.idle: return

            while self.writes_remain():
                for cax in caxs: cax.clear()
                dsets, ni = self.get_dsets(tasks)
                assert self.current_file_handle is not None, "current_file_handle must be set"
                sim_time = self.current_file_handle['scales/sim_time'][ni]
                write_num = self.current_file_handle['scales/write_number'][ni]
                for k, cm in self.colormeshes:
                    ax = axs[k]
                    cax = caxs[k]
                    cm.plot_colormesh(ax, cax, dsets[cm.task], ni, **mpl_kwargs)
                plt.suptitle('t = {:.4e}'.format(sim_time))
                self.grid.fig.savefig('{:s}/{:s}_{:06d}.png'.format(self.out_dir, self.out_name, int(write_num+start_fig-1)), dpi=dpi, bbox_inches='tight')
                
                for k, cm in self.colormeshes:
                    axs[k].cla()
                    caxs[k].cla()

