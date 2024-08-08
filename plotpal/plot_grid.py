from typing import Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # type: ignore

class PlotGrid:
    """ Creates a grid of matplotlib subplots with a specified number of rows and columns. """
    
    def __init__(self, col_inch: float = 3, row_inch: float = 3, pad_factor: float = 10):
        """
        Initialize a PlotGrid object.

        Parameters
        ----------
        col_inch : The width of each column in inches.
        row_inch : The height of each row in inches.
        pad_factor : The amount of padding between subplots as a percentage of the total width/height of the subplot.
        """
        self.specs: list[dict[str, Any]] = []
        self.col_inch = col_inch
        self.row_inch = row_inch
        self.pad_factor = pad_factor
        self.nrows: int = 0
        self.ncols: int = 0
        self.axes: dict[str, matplotlib.axes.Axes] = {}
        self.cbar_axes: dict[str, matplotlib.axes.Axes] = {}

    def add_axis(self, 
                 row_num: int,
                 col_num: int, 
                 row_span: int = 1, 
                 col_span: int = 1, 
                 cbar: bool = False, 
                 polar: bool = False, 
                 mollweide: bool = False, 
                 orthographic: bool = False, 
                 threeD: bool = False
                 ) -> None:
        """
        Add an axis to the grid.
        
        Parameters
        ----------
        row_num : The row number of the axis. The first row is 0.
        col_num : The column number of the axis. The first column is 0.
        row_span : The number of rows that this subplot axis spans.
        col_span : The number of columns that this subplot axis spans.
        cbar : Whether or not to add a colorbar to this axis.
        polar : Whether or not to use a polar projection for this axis.
        mollweide : Whether or not to use a mollweide projection for this axis.
        orthographic : Whether or not to use an orthographic projection for this axis.
        threeD : Whether or not to use a 3D projection for this axis.
        """
        if sum([polar, mollweide, orthographic, threeD]) > 1:
            raise ValueError("Only one of polar, mollweide, orthographic, or threeD can be True.")
        
        subplot_kwargs: dict[str, Any] = {'polar' : polar}
        if mollweide:
            subplot_kwargs['projection'] = 'mollweide'
        elif orthographic:
            subplot_kwargs['projection'] = ccrs.Orthographic(180, 45)
        elif threeD:
            subplot_kwargs['projection'] = '3d'

        this_spec: dict[str, Any] = {}
        this_spec['row_num'] = row_num
        this_spec['col_num'] = col_num
        this_spec['row_span'] = row_span
        this_spec['col_span'] = col_span
        this_spec['cbar'] = cbar
        this_spec['kwargs'] = subplot_kwargs
        self.specs.append(this_spec)

    def make_subplots(self) -> None:
        """ Generates the subplots. """

        # If the user oopsied on their specs and added a subplot that goes off the grid, fix it.
        for spec in self.specs:
            row_ceil = int(np.ceil(spec['row_num'] + spec['row_span']))
            col_ceil = int(np.ceil(spec['col_num'] + spec['col_span']))
            if row_ceil > self.nrows:
                self.nrows = row_ceil
            if col_ceil > self.ncols:
                self.ncols = col_ceil
        self.fig = plt.figure(figsize=(self.ncols*self.col_inch, self.nrows*self.row_inch))

        # fractional width and height of each subplot
        x_factor = 1/self.ncols
        y_factor = 1/self.nrows

        # fractional offset of each subplot from the left and bottom edges
        x_offset = 0.5*x_factor*self.pad_factor/100
        y_offset = 0

        for spec in self.specs:
            col_spot = spec['col_num']
            row_spot = self.nrows - spec['row_num'] - 1

            #anchor = (x,y) of lower left corner of subplot
            x_anchor = col_spot*x_factor + x_offset
            y_anchor = row_spot*y_factor + y_offset
            delta_x = spec['col_span']*x_factor * (1 - self.pad_factor/100)
            delta_y = spec['row_span']*y_factor * (1 - self.pad_factor/100)

            if spec['cbar']:
                # If the user wants a colorbar, make room for it.
                if spec['kwargs']['polar']:
                    cbar_y_anchor = y_anchor + 0.95*delta_y
                    cbar_x_anchor = x_anchor + 0.1*delta_x
                    cbar_delta_y = 0.05*delta_y
                    cbar_delta_x = 0.15*delta_x
                    delta_y *= 0.95
                else:
                    cbar_y_anchor = y_anchor + 0.9*delta_y
                    cbar_x_anchor = x_anchor + 0.1*delta_x
                    cbar_delta_y = 0.1*delta_y
                    cbar_delta_x = 0.15*delta_x
                    delta_y *= 0.85
                self.cbar_axes['ax_{}-{}'.format(spec['row_num'], spec['col_num'])] = self.fig.add_axes((cbar_x_anchor, cbar_y_anchor, cbar_delta_x, cbar_delta_y))
            self.axes['ax_{}-{}'.format(spec['row_num'], spec['col_num'])] = self.fig.add_axes((x_anchor, y_anchor, delta_x, delta_y), **spec['kwargs'])


class RegularPlotGrid(PlotGrid):
    """ Makes a grid of subplots where each plot spans exactly one row and one column and each axis has the same projection. """

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
            pad_factor: float = 10
            ):
        self.num_rows     = num_rows
        self.num_cols     = num_cols
        super().__init__(col_inch=col_inch, row_inch=row_inch, pad_factor=pad_factor)

        for i in range(num_rows):
            for j in range(num_cols):
                self.add_axis(row_num=i, col_num=j, row_span=1, col_span=1, cbar=cbar, polar=polar, mollweide=mollweide, orthographic=orthographic, threeD=threeD)
        self.make_subplots()

def RegularColorbarPlotGrid(*args, **kwargs): #type: ignore
    kwargs['cbar'] = True
    return RegularPlotGrid(*args, **kwargs)

class PyVista3DPlotGrid:
    """
    A class for making a grid of 3D plots using PyVista
    """

    def __init__(self, num_rows: int = 1, num_cols: int = 1, size: int = 500):
        """
        Initialize the grid of plots

        Parameters
        ----------
        num_rows : int
            Number of rows in the grid
        num_cols : int
            Number of columns in the grid
        size : int
            Size of each subplot in pixels
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("PyVista must be installed for 3D pyvista plotting in plotpal")

        self.pl = pv.Plotter(off_screen=True, shape=(num_rows, num_cols)) # type: ignore
        self.num_rows     = num_rows   
        self.num_cols     = num_cols   
        self.size = size

    def change_focus(self, row: int, col: int) -> None:
        """ Focus on a particular plot in the grid; row and col are 0-indexed """
        self.pl.subplot(row, col) # type: ignore
    
    def change_focus_single(self, index: int) -> None:
        """ Focus on a particular plot in the grid; indexed from left to right, top to bottom """
        row = index // self.num_cols
        col = index % self.num_cols
        self.change_focus(row, col) 

    def save(self, filename: str) -> None:
        self.pl.screenshot(filename=filename, window_size=[self.num_cols*self.size, self.num_rows*self.size]) # type: ignore
