import logging
from collections import OrderedDict
from sys import stdout
from typing import Optional

import numpy as np
import h5py #type: ignore
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator #type: ignore
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 9})

from plotpal.file_reader import SingleTypeReader, match_basis
from plotpal.plot_grid import RegularPlotGrid

logger = logging.getLogger(__name__.split('.')[-1])


class PdfPlotter(SingleTypeReader):
    """
    A class for plotting probability distributions of a dedalus output task.

    PDF plots are currently implemented for 2D slices and 3D volumes. 
    When one axis is represented by polynomials that exist on an uneven basis (e.g., Chebyshev),
    that basis is evenly interpolated to avoid skewing of the distribution by uneven grid sampling.
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
        Initializes the PDF plotter.
        """
        super(PdfPlotter, self).__init__(
            run_dir=run_dir,
            sub_dir=sub_dir,
            out_name=out_name,
            distribution=distribution,
            num_files=num_files,
            roll_writes=roll_writes,
            start_file=start_file,
            global_comm=global_comm,
            chunk_size=chunk_size
        )
        self.pdfs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self.pdf_stats: dict[str, tuple[float, float, float, float]] = OrderedDict()

    def _calculate_pdf_statistics(self) -> None:
        """ Calculate statistics of the PDFs stored in self.pdfs. Store results in self.pdf_stats. """
        for k, data in self.pdfs.items():
            pdf, x_vals, dx = data

            mean = np.sum(x_vals*pdf*dx)
            stdev = np.sqrt(np.sum((x_vals-mean)**2*pdf*dx))
            skew = np.sum((x_vals-mean)**3*pdf*dx)/stdev**3
            kurt = np.sum((x_vals-mean)**4*pdf*dx)/stdev**4
            self.pdf_stats[k] = (mean, stdev, skew, kurt)

    
    def _get_interpolated_slices(
            self, 
            dsets: dict[str, h5py.Dataset],
            ni: int, 
            uneven_basis: Optional[str] = None
            ) -> dict[str, np.ndarray]:
        """
        For 2D data on an uneven grid, interpolates that data on to an evenly spaced grid.

        # Arguments
            dsets : A dictionary of links to dedalus output tasks in hdf5 files.
            ni : The index of the slice to be interpolate.
            uneven_basis : The basis on which the grid has uneven spacing.
        """
        #Read data
        bases = self.current_bases

        # Create an even x- and y- grid
        x, y = [match_basis(dsets[next(iter(dsets))], bn) for bn in bases]
        if bases[0] == uneven_basis:
            even_x = np.linspace(x.min(), x.max(), len(x))
            even_y = y
        elif bases[1] == uneven_basis:
            even_x = x
            even_y = np.linspace(y.min(), y.max(), len(y))
        else:
            even_x, even_y = x, y
        eyy, exx = np.meshgrid(even_y, even_x)

        # Interpolate data onto the even grid
        file_data = OrderedDict()
        for k in dsets.keys(): 
            file_data[k] = np.zeros(dsets[k][ni].shape)
            interp = RegularGridInterpolator((x.flatten(), y.flatten()), dsets[k][ni], method='linear')
            file_data[k][:,:] = interp((exx, eyy))

        return file_data

    def _get_interpolated_volumes(
            self, 
            dsets: dict[str, h5py.Dataset],
            ni: int, 
            uneven_basis: Optional[str] = None
            ) -> dict[str, np.ndarray]:
        """
        For 3D data on an uneven grid, interpolates that data on to an evenly spaced grid.

        # Arguments
            dsets : A dictionary of links to dedalus output tasks in hdf5 files.
            ni : The index of the field to be interpolate.
            uneven_basis : The basis on which the grid has uneven spacing.
        """
        #Read data
        bases = self.current_bases

        # Create an even x-, y-, and z- grid
        x, y, z = [match_basis(dsets[next(iter(dsets))], bn) for bn in bases]
        uneven_index  = None
        if bases[0] == uneven_basis:
            even_x = np.linspace(x.min(), x.max(), len(x))
            even_y = y
            even_z = z
            uneven_index = 0
        elif bases[1] == uneven_basis:
            even_x = x
            even_y = np.linspace(y.min(), y.max(), len(y))
            even_z = z
            uneven_index = 1
        elif bases[2] == uneven_basis:
            even_x = x
            even_y = y
            even_z = np.linspace(z.min(), z.max(), len(z))
            uneven_index = 2
        else:
            even_x, even_y, even_z = x, y, z

        exx, eyy, ezz = None, None, None
        if uneven_index == 0:
            eyy, exx = np.meshgrid(even_y, even_x)
        elif uneven_index == 1:
            eyy, exx = np.meshgrid(even_y, even_x)
        elif uneven_index == 2:
            ezz, exx = np.meshgrid(even_z, even_x)

        file_data = OrderedDict()

        # Interpolate data onto the even grid
        #TODO: Double-check logic here -- this is years old.
        for k in dsets.keys(): 
            file_data[k] = np.zeros(dsets[k][ni].shape)
            for i in range(file_data[k].shape[0]):
                if self.comm.rank == 0:
                    print('interpolating {} ({}/{})...'.format(k, i+1, file_data[k].shape[0]))
                    stdout.flush()
                if uneven_index is None:
                    file_data[k][i,:] = dsets[k][ni][i,:]
                elif uneven_index == 2:
                    for j in range(file_data[k].shape[-2]): # loop over y
                        interp = RegularGridInterpolator((x.flatten(), z.flatten()), dsets[k][i,:,j,:], method='linear')
                        file_data[k][i,:,j,:] = interp((exx, ezz))
                else:
                    for j in range(file_data[k].shape[-1]): # loop over z
                        interp = RegularGridInterpolator((x.flatten(), y.flatten()), dsets[k][i,:,:,j], method='linear')
                        file_data[k][i,:,:,j] = interp((exx, eyy))

        return file_data

    def _get_bounds(self, pdf_list: list[str]) -> dict[str, np.ndarray]:
        """
        Finds the global minimum and maximum value of fields for determing PDF range.

        Arguments
        ---------
        pdf_list : A list of fields for which to calculate the global minimum and maximum.
        """    
        with self.my_sync:
            if self.idle : return {}

            bounds = OrderedDict()
            for field in pdf_list:
                bounds[field] = np.zeros(2)
                bounds[field][:] = np.nan

            # Find the local minimum and maximum
            while self.writes_remain():
                dsets, ni = self.get_dsets(pdf_list)
                for field in pdf_list:
                    if np.isnan(bounds[field][0]):
                        bounds[field][0], bounds[field][1] = dsets[field][ni].min(), dsets[field][ni].max()
                    else:
                        if dsets[field][ni].min() < bounds[field][0]:
                            bounds[field][0] = dsets[field][ni].min()
                        if dsets[field][ni].max() > bounds[field][1]:
                            bounds[field][1] = dsets[field][ni].max()

            # Communicate the global minimum and maximum
            for field in pdf_list:
                buff     = np.zeros(1)
                self.comm.Allreduce(bounds[field][0], buff, op=MPI.MIN)
                bounds[field][0] = buff

                self.comm.Allreduce(bounds[field][1], buff, op=MPI.MAX)
                bounds[field][1] = buff

            return bounds


    def calculate_pdfs(
            self, 
            pdf_list: list[str], 
            bins: int = 100, 
            threeD: bool = False, 
            bases: list[str]=['x', 'z'], 
            uneven_basis: Optional[str] = None,
            ) -> None:
        """
        Calculate probability distribution functions of the specified tasks.

        # Arguments
            pdf_list (list) :
                The names of the tasks to create PDFs of
            bins (int, optional) :
                The number of bins the PDF (histogram) should have
            threeD (bool, optional) :
                If True, find PDF of a 3D volume
            bases : A list of strings of the bases over which the simulation information spans. 
                Should have 2 elements if threeD is False, 3 elements if threeD is True.
            uneven_basis : The basis on which the grid has uneven spacing, if any.
        """
        self.current_bases = bases
        bounds = self._get_bounds(pdf_list)

        histograms = OrderedDict()
        bin_edges  = OrderedDict()
        for field in pdf_list:
            histograms[field] = np.zeros(bins)
            bin_edges[field] = np.zeros(bins+1)

        with self.my_sync:
            if self.idle : return

            while self.writes_remain():
                dsets, ni = self.get_dsets(pdf_list)

                # Interpolate data onto a regular grid
                if threeD:
                    file_data = self._get_interpolated_volumes(dsets, ni, uneven_basis=uneven_basis)
                else:
                    file_data = self._get_interpolated_slices(dsets, ni, uneven_basis=uneven_basis)

                # Create histograms of data
                for field in pdf_list:
                    hist, bin_vals = np.histogram(file_data[field], bins=bins, range=tuple(bounds[field]))
                    histograms[field] += hist
                    bin_edges[field] = bin_vals


            for field in pdf_list:
                # Communicate the global histogram (counts per bin)
                loc_hist    = np.array(histograms[field], dtype=np.float64)
                global_hist = np.zeros_like(loc_hist, dtype=np.float64)
                self.comm.Allreduce(loc_hist, global_hist, op=MPI.SUM)

                # Calculate the PDF from the histogram
                dx = bin_edges[field][1]-bin_edges[field][0]
                x_vals  = bin_edges[field][:-1] + dx/2
                pdf     = global_hist/np.sum(global_hist)/dx
                self.pdfs[field] = (pdf, x_vals, dx)

            self._calculate_pdf_statistics()
        

    def plot_pdfs(
            self, 
            dpi: int = 150, 
            col_inch: float = 3, 
            row_inch: float = 3, 
            ) -> None:
        """
        Plot the probability distribution functions and save them to file.

        # Arguments
            dpi :  Pixel density of output image.
            col_inch : Width of each column in inches.
            row_inch : Height of each row in inches.
        """
        with self.my_sync:
            if self.comm.rank != 0: return

            grid = RegularPlotGrid(
                num_rows=1,
                num_cols=1, 
                cbar=False,
                polar=False,
                mollweide=False,
                orthographic=False,
                threeD=False,
                col_inch=col_inch,
                row_inch=row_inch,
                pad_factor=10
                )
            ax = grid.axes['ax_0-0']
            
            for k, data in self.pdfs.items():
                pdf, xs, dx = data
                mean, stdev, skew, kurt = self.pdf_stats[k]
                title = r'$\mu$ = {:.2g}, $\sigma$ = {:.2g}, skew = {:.2g}, kurt = {:.2g}'.format(mean, stdev, skew, kurt)
                ax.set_title(title)
                ax.axvline(mean, c='orange')

                ax.plot(xs, pdf, lw=2, c='k')
                ax.fill_between((mean-stdev, mean+stdev), pdf.min(), pdf.max(), color='orange', alpha=0.5)
                ax.fill_between(xs, 1e-16, pdf, color='k', alpha=0.5)
                ax.set_xlim(xs.min(), xs.max())
                ax.set_ylim(pdf[pdf > 0].min(), pdf.max())
                ax.set_yscale('log')
                ax.set_xlabel(k)
                ax.set_ylabel('P({:s})'.format(k))

                grid.fig.savefig('{:s}/{:s}_pdf.png'.format(self.out_dir, k), dpi=dpi, bbox_inches='tight')
                ax.clear()

            self._save_pdfs()

    def _save_pdfs(self) -> None:
        """ 
        Save PDFs to file. For each PDF, e.g., 'entropy' and 'w', the file will have a dataset with:
            xs  - the x-values of the PDF
            pdf - the (normalized) y-values of the PDF
            dx  - the spacing between x values, for use in integrals.
        """
        if self.comm.rank == 0:
            with h5py.File('{:s}/pdf_data.h5'.format(self.out_dir), 'w') as f:
                for k, data in self.pdfs.items():
                    pdf, xs, dx = data
                    this_group = f.create_group(k)
                    for d, n in ((pdf, 'pdf'), (xs, 'xs')):
                        this_group.create_dataset(name=n, shape=d.shape, dtype=np.float64)
                        f['{:s}/{:s}'.format(k, n)][:] = d
                    this_group.create_dataset(name='dx', shape=(1,), dtype=np.float64)
                    f['{:s}/dx'.format(k)][0] = dx

