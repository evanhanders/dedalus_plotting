import logging
from collections import OrderedDict
from typing import Optional

import numpy as np
import h5py #type: ignore
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as ticker
matplotlib.rcParams.update({'font.size': 9})
from mpi4py import MPI

from plotpal.file_reader import SingleTypeReader, RolledDset
from plotpal.plot_grid import RegularPlotGrid

logger = logging.getLogger(__name__.split('.')[-1])


class ScalarFigure(RegularPlotGrid):
    """
    A simple extension of the RegularPlotGrid class tailored specifically for scalar line traces.

    Scalar traces are put on panels, which are given integer indices.
    Panel 0 is the axis subplot to the upper left, and panel indices increase to the
    right, and downwards, like reading a book.
    """

    def __init__(
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
            pad_factor: float = 10,
            fig_name: Optional[str] = None
            ):
        super(ScalarFigure, self).__init__(
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
        self.panels: list[str] = []
        self.panel_fields: list[list[tuple[str, bool, dict]]] = []
        self.fig_name = fig_name
        self.color_ind: list[int] = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.panels.append('ax_{}-{}'.format(i,j))
                self.panel_fields.append([])
                self.color_ind.append(0)

    def add_field(
            self, 
            panel: int, 
            field: str, 
            log: bool = False, 
            extra_kwargs: dict = {}
            ) -> None:
        """
        Add a field to a specified panel

        # Arguments
            panel : The panel index to add this field to
            field : The name of the dedalus task to plot on this panel
            log : If True, log-scale the y-axis of the plot
            extra_kwargs : Any extra keyword arguments to pass to the plot function
        """
        if 'c' not in extra_kwargs and 'color' not in extra_kwargs:
            extra_kwargs['color'] = 'C{}'.format(self.color_ind[panel])
            self.color_ind[panel] += 1
        self.panel_fields[panel].append((field, log, extra_kwargs))


class ScalarPlotter(SingleTypeReader):
    """
    A class for plotting traces of scalar values from dedalus output.
    """

    def __init__(
            self,
            run_dir: str, 
            sub_dir: str, 
            out_name: str, 
            distribution: str = 'single',
            num_files: Optional[int] = None, 
            roll_writes: Optional[int] = None,
            start_file: int = 1,
            global_comm: MPI.Intracomm = MPI.COMM_WORLD,
            chunk_size: int = 1000
            ):
        super().__init__(
            run_dir=run_dir,
            sub_dir=sub_dir,
            out_name=out_name,
            distribution=distribution,
            num_files=num_files,
            roll_writes=None,
            start_file=start_file,
            global_comm=global_comm,
            chunk_size=chunk_size
            )
        if roll_writes is not None:
            self.rolled_reader: Optional[SingleTypeReader] = SingleTypeReader(
            run_dir=run_dir,
            sub_dir=sub_dir,
            out_name=out_name,
            distribution=distribution,
            num_files=num_files,
            roll_writes=None,
            start_file=start_file,
            global_comm=global_comm,
            chunk_size=chunk_size
            )
        else:
            self.rolled_reader = None
        self.fields: list[str] = []
        self.trace_data: dict[str, np.ndarray] = OrderedDict()

    def load_figures(self, fig_list: list[ScalarFigure]) -> None:
        """
        Loads a list of ScalarFigure objects and parses them to see which fields must be read from file.

        # Arguments
            fig_list : The ScalarFigure objects to be plotted.
        """
        self.figures = fig_list
        for fig in self.figures:
            for field_list in fig.panel_fields:
                for fd, _, _ in field_list:
                    if fd not in self.fields:
                        self.fields.append(fd)

    def _read_fields(self) -> None:
        """ Reads scalar data from file """
        with self.my_sync:
            if self.idle: return

            # Make space for the data
            trace_data: dict[str, list] = OrderedDict()
            for f in self.fields: 
                trace_data[f] = []
                if self.rolled_reader is not None:
                    trace_data['rolled_{}'.format(f)] = []
            trace_data['sim_time'] = []

            # Read the data (currently write-by-write; could be improved but scalars are small so shrug)
            while self.writes_remain():
                assert isinstance(self.current_file_handle, h5py.File), "current_file_handle is not an h5py.File"
                dsets, ni, rdsets, ri = self.get_dsets(self.fields)
                for f in self.fields: 
                    trace_data[f].append(dsets[f][ni].squeeze())
                    if rdsets is not None and ri is not None:
                        trace_data['rolled_{}'.format(f)].append(rdsets[f][ri].squeeze())
                
                trace_data['sim_time'].append(self.current_file_handle['scales']['sim_time'][ni])

            for f in self.fields: self.trace_data[f] = np.array(trace_data[f])
            self.trace_data['sim_time'] = np.array(trace_data['sim_time'])

    def _clear_figures(self) -> None:
        """ Clear the axes on all figures """
        for f in self.figures:
            for _, k in enumerate(f.panels): 
                f.axes[k].clear()

    def _save_traces(self) -> None:
        """ save traces to file """
        if self.idle:
            return
        with h5py.File('{:s}/full_traces.h5'.format(self.out_dir), 'w') as f:
            for k, fd in self.trace_data.items():
                f[k] = fd

    def get_dsets(self, *args, **kwargs) -> tuple[dict[str, h5py.Dataset], int, Optional[dict[str, RolledDset]], Optional[int]]: #type: ignore
        """ A wrapper for the parent class's get_dsets method that also returns the rolled reader's dsets if it exists"""
        dsets, ni = super().get_dsets(*args, **kwargs)
        rolled_dsets: Optional[dict[str, RolledDset]] = None
        ri : Optional[int] = None
        if self.rolled_reader is not None:
            rolled_dsets, ri = self.rolled_reader.get_dsets(*args, **kwargs)
        return dsets, ni, rolled_dsets, ri

    def writes_remain(self) -> bool:
        """ A wrapper for the parent class's writes_remain method that also returns the rolled reader's writes_remain if it exists"""
        if self.rolled_reader is None:
            return super().writes_remain()
        else:
            return super().writes_remain() and self.rolled_reader.writes_remain()

    def plot_figures(self, dpi: int = 200, fig_name: str = 'output') -> None:
        """ 
        Plot scalar traces vs. time

        # Arguments
            dpi : image pixel density
            fig_name : default name of the output figure (if not specified in the ScalarFigure object)
        """
        with self.my_sync:
            if self.trace_data is None:
                self._read_fields()
            self._clear_figures()
            if self.idle: return

            for j, fig in enumerate(self.figures):
                for i, k in enumerate(fig.panels):
                    ax = fig.axes[k]
                    for fd, log, kwargs in fig.panel_fields[i]:
                        ax.plot(self.trace_data['sim_time'], self.trace_data[fd], label=fd, **kwargs)
                        if self.rolled_reader is not None:
                            if 'lw' not in kwargs and 'linewidth' not in kwargs:
                                kwargs['lw'] = 2
                            ax.plot(self.trace_data['sim_time'], self.trace_data['rolled_{}'.format(fd)], label='rolled_{}'.format(fd), **kwargs)
                        if log:
                            ax.set_yscale('log')
                    ax.set_xlim(self.trace_data['sim_time'].min(), self.trace_data['sim_time'].max())
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
                    ax.legend(fontsize=8, loc='best')
                ax.set_xlabel('sim_time')
                if fig.fig_name is None:
                    fig_name = fig_name + '_{}'.format(j)
                else:
                    fig_name = fig.fig_name

                fig.fig.savefig('{:s}/{:s}.png'.format(self.out_dir, fig_name), dpi=dpi, bbox_inches='tight')
            self._save_traces()

    def plot_convergence_figures(self, dpi: int = 200, fig_name: str = 'output') -> None:
        """ 
        Plot scalar convergence traces vs. time
        Plotted is fractional difference of the value at a given time compared to the final value:

        abs( 1 - time_trace/final_value), 

        where final_value is the mean value of the last 10% of the trace data

        # Arguments
            dpi : image pixel density
            fig_name : default name of the output figure (if not specified in the ScalarFigure object)
        """

        with self.my_sync:
            if self.trace_data is None:
                self._read_fields()
            self._clear_figures()
            if self.idle: return

            for j, fig in enumerate(self.figures):
                for i, k in enumerate(fig.panels):
                    ax = fig.axes[k]
                    ax.grid(which='major')
                    for fd, log, kwargs in fig.panel_fields[i]:
                        these_kwargs = kwargs.copy()
                        if 'rolled_{}'.format(fd) in self.trace_data:
                            final_mean = self.trace_data['rolled_{}'.format(fd)][-1]
                            these_kwargs['label'] = "1 - ({:s})/(final rolled mean)".format(fd)
                        else:
                            final_mean = np.mean(self.trace_data[fd][-int(0.1*len(self.trace_data[fd])):])
                            these_kwargs['label'] = "1 - ({:s})/(last 10% mean)".format(fd)
                        ax.plot(self.trace_data['sim_time'], np.abs(1 - self.trace_data[fd]/final_mean), **these_kwargs)
                    ax.set_yscale('log')
                    ax.set_xlim(self.trace_data['sim_time'].min(), self.trace_data['sim_time'].max())
                    ax.legend(fontsize=8, loc='best')
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
                ax.set_xlabel('sim_time')
                if fig.fig_name is None:
                    fig_name = fig_name + '_{}'.format(j)
                else:
                    fig_name = fig.fig_name

                fig.fig.savefig('{:s}/{:s}_convergence.png'.format(self.out_dir, fig_name), dpi=dpi, bbox_inches='tight')
