#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:22:17 2022

@author: dejan
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import (Slider, Button, RadioButtons, SpanSelector,
                                CheckButtons)
from skimage import draw
import calculate as cc


def set_img_coordinates(da, ax, unit="µm",
                        rowcoord_arr=None, colcoord_arr=None):

    if rowcoord_arr is None:
        rowcoord_arr = np.unique(da[da.RowCoord].data)
    if colcoord_arr is None:
        colcoord_arr = np.unique(da[da.ColCoord].data)

    def row_coord(y, pos):
        yind = int(y)
        if yind <= len(rowcoord_arr):
            yy = f"{rowcoord_arr[yind]}{unit}"
        else:
            yy = ""
        return yy

    def col_coord(x, pos):
        xind = int(x)
        if xind < len(colcoord_arr):
            xx = f"{colcoord_arr[xind]}{unit}"
        else:
            xx = ""
        return xx

    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(col_coord))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(row_coord))
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False,
                   width=.2, labelsize=8)

class ShowSelected(object):
    """Select a span and plot a map of a chosen function in that span.
    Right-Click (or middle-click) on the image to see the spectrum
    corresponding to that pixel.

    To be used for visual exploration of the maps.
    The lower part of the figure contains the spectra you can scroll through
    using the slider just beneath the spectra.
    You can use your mouse to select a zone in the spectra and a map plot
    should appear in the upper part of the figure.
    On the left part of the figure you can select what kind of function
    you want to apply on the selected span.
    ['area', 'barycenter_x', 'max_value', 'peak_position', 'peak_ratio']
    """

    def __init__(self, input_spectra, x=None, interpolation='gaussian', **kwargs):

        self.da = input_spectra.copy()
        self.interp = interpolation
        self.f_names = ['area',
                        'barycenter_x',
                        'max_value',
                        'peak_position',
                        'peak_ratio']
        self.xmin = self.da.shifts.data.min()
        self.xmax = self.da.shifts.data.max()
        # Get some basic info on the spectra:
        self.nshifts = self.da.attrs["PointsPerSpectrum"]
        self.ny, self.nx = self.da.attrs["ScanShape"]
        self.scan_type = self.da.attrs["MeasurementType"]
        self.cmap = kwargs.pop("cmap", "viridis")
        self.file_location = kwargs.pop("file_location", "./")
        self.extent = kwargs.pop("extent", "full")
        #---------------------------- about labels ---------------------------#
        try:
            xlabel = self.da.attrs["ColCoord"]
            ylabel = self.da.attrs["RowCoord"]
            if (self.scan_type == "Map") and (self.da.MapAreaType != "Slice"):
                self.xlabel = f"{xlabel} [{input_spectra[xlabel].units}]"
                self.ylabel = f"{ylabel} [{input_spectra[ylabel].units}]"
            else: # Not a map scan
                self.xlabel = xlabel
                self.ylabel = ylabel
        except:
            self.xlabel = "X"
            self.ylabel = "Y"
        #---------------------------------------------------------------------#

        # Preparing the plots:
        figsize = kwargs.pop("figsize", (14, 8))
        self.fig = plt.figure(figsize=figsize, **kwargs)
        # Add all the axes:
        self.aximg = self.fig.add_axes([.21, .3, .74, .6]) # main frame
        self.axspectrum = self.fig.add_axes([.05, .075, .9, .15])
        self.axradio = self.fig.add_axes([.05, .3, .1, .6])
        self.axreduce = self.fig.add_axes([.05, .275, .1, .05])
        self.axabsscale = self.fig.add_axes([.05, .22, .1, .05])
        self.axsave = self.fig.add_axes([.05, .905, .1, .08])
        self.axscroll = self.fig.add_axes([.05, .02, .9, .02])
        self.axradio.axis('off')
        self.axreduce.axis('off')
        self.axabsscale.axis('off')
        # self.axsave.axis('off')
        self.axscroll.axis('off')
        self.first_frame = 0
        if self.scan_type != "Single":
            # Slider to scroll through spectra:
            self.last_frame = len(self.da.data)-1
            self.sframe = Slider(self.axscroll, 'S.N°',
                                 self.first_frame, self.last_frame,
                                 valinit=self.first_frame, valfmt='%d', valstep=1)
            self.sframe.on_changed(self.scroll_spectra)
        # Show the spectrum:
        self.spectrumplot, = self.axspectrum.plot(self.da.shifts.data,
                                                  self.da.data[self.first_frame])
        self.titled(self.axspectrum, self.first_frame)
        self.vline = None
        self.func = "max"  # Default function
        self.xmin = self.da.shifts.data.min()
        self.xmax = self.da.shifts.data.max()
        self.span = SpanSelector(self.axspectrum, self.onselect, 'horizontal',
                                 useblit=True, span_stays=True,
                                 rectprops=dict(alpha=0.5,
                                                facecolor='tab:blue'))
        self.func_choice = RadioButtons(self.axradio, self.f_names)
        self.func_choice.on_clicked(self.determine_func)
        self.reduced_button = CheckButtons(self.axreduce, ["reduced"])
        self.reduced_button.on_clicked(self.is_reduced)
        self.reduced = self.reduced_button.get_status()[0]
        self.abs_scale_button = CheckButtons(self.axabsscale, ["abs. scale"])
        self.abs_scale_button.on_clicked(self.is_absolute_scale)
        self.absolute_scale = self.abs_scale_button.get_status()[0]
        self.save_button = Button(self.axsave, "Save Image")
        self.save_button.on_clicked(self.save_image)
        self.func = self.func_choice.value_selected
        # Plot the empty "image":
        if self.scan_type == "Map":
            self.imup = self.aximg.imshow(cc.calculate_ss(
                                        self.func,
                                        self.da),
                                        interpolation=self.interp,
                                        aspect=self.nx/self.ny,
                                        cmap=self.cmap)
            self.aximg.set_xlabel(f"{self.xlabel}")
            self.aximg.xaxis.set_label_position('top')
            self.aximg.set_ylabel(f"{self.ylabel}")
            try:
                set_img_coordinates(self.da, self.aximg, unit="")
            except:
                pass
            self.cbar = self.fig.colorbar(self.imup, ax=self.aximg)

        elif self.scan_type == 'Single':
            self.aximg.axis('off')
            self.imup = self.aximg.annotate('calculation result', (.4, .8),
                                        style='italic', fontsize=14,
                                        xycoords='axes fraction',
                                        bbox={'facecolor': 'lightcoral',
                                        'alpha': 0.3, 'pad': 10})
        else:
            _length = np.max((self.ny, self.nx))
            self.imup, = self.aximg.plot(np.arange(_length),
                                         np.zeros(_length), '--o', alpha=.5)

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        # self.fig.show()
    def is_reduced(self, label):
        self.reduced = self.reduced_button.get_status()[0]
        self.draw_img()

    def is_absolute_scale(self, label):
        self.absolute_scale = self.abs_scale_button.get_status()[0]
        self.draw_img()

    # def increment_filename(self, filename):

    #     if not path.exists(filename):
    #         return filename
    #     file, extension = path.splitext(filename)
    #     counter = file[-3:]
    #     if counter.isnumeric():
    #         file = file[:-3]
    #         counter = f"{int(counter) + 1:03d}"
    #     else:
    #         counter = "001"

    #     filename = file + counter + extension
    #     return self.increment_filename(filename)

    def full_extent(self, padx=0.2, pady=0.0):
        """Get the full extent of an axes, including axes labels, tick labels, and
        titles."""

        # ax.figure.canvas.draw()
        items = self.aximg.get_xticklabels() + self.aximg.get_yticklabels()
        items += [self.aximg, self.aximg.title, self.aximg.xaxis.label, self.aximg.yaxis.label]
        items += [self.aximg.title]
        bbox = mpl.transforms.Bbox.union([item.get_window_extent() for item in items])
        dx, dy = self.aximg.transAxes.transform((0.15, 0.025)) -\
                 self.aximg.transAxes.transform((0, 0))
        return bbox.expanded(1.0 + padx, 1.0 + pady).translated(dx, dy)

    # def save_image(self, event):
    #     filename = path.join(self.file_location, self.func+"_000.png")
    #     figsize = self.fig.get_size_inches()
    #     self.fig.set_size_inches(16, 10, forward=False)
    #     if self.extent == "full":
    #         myextent = self.full_extent().transformed(self.fig.dpi_scale_trans.inverted())
    #     elif self.extent == "tight":
    #         myextent = self.aximg.get_tightbbox(self.fig.canvas.renderer).transformed(self.fig.dpi_scale_trans.inverted())
    #     else:
    #         warn("The 'extent' keyword must be one of [\"full\", \"tight\"]")
    #     savename = self.increment_filename(filename)
    #     # print(savename)
    #     self.fig.savefig(savename, bbox_inches=myextent, transparent=True, dpi=120)
    #     self.fig.set_size_inches(figsize)


    def onclick(self, event):
        """Right-Clicking on a pixel will show the spectrum
        corresponding to that pixel on the bottom plot"""
        if event.inaxes == self.aximg:
            if event.button != 1:
                x_pos = round(event.xdata)
                y_pos = round(event.ydata)
                if isinstance(self.imup, mpl.image.AxesImage): # if image
                    if x_pos <= self.nx and y_pos <= self.ny and x_pos * y_pos >= 0:
                        broj = round(y_pos * self.nx + x_pos)
                        self.sframe.set_val(broj)
                        self.scroll_spectra(broj)
                elif isinstance(self.imup, mpl.lines.Line2D):
                    broj = x_pos
                    self.sframe.set_val(broj)
                    self.scroll_spectra(broj)
            else:
                pass

    def determine_func(self, label):
        "Recover the function name from radio button clicked"""
        self.func = label
        self.draw_img()

    def onselect(self, xmin, xmax):
        """When you select a region of the spectra."""
        self.xmin = xmin
        self.xmax = xmax
        if self.vline:
            self.axspectrum.lines.remove(self.vline)
            self.vline = None
        self.draw_img()

    def draw_img(self):
        """Draw/update the image."""
        # calculate the function:
        whole_data = cc.calculate_ss(self.func, self.da, self.xmin, self.xmax,
                                     is_reduced=self.reduced)
        if self.scan_type == 'Single':
            naj = 1.
        else:
            naj = np.max(whole_data)
        img = whole_data / naj
        if self.scan_type == "Map":
            self.imup.set_data(img)
            limits = np.percentile(img, [1, 99])
            if self.absolute_scale:
                limits = [0, img.max()]
            self.imup.set_clim(limits)
            self.cbar.mappable.set_clim(*limits)
        elif self.scan_type == 'Single':
            self.imup.set_text(f"{img[0][0]:.3G}")
        else:
            self.imup.set_ydata(img.squeeze())
            self.aximg.relim()
            self.aximg.autoscale_view(None, False, True)

        self.aximg.set_title(f"Calculated {'reduced'*self.reduced} {self.func} "
                             f"between {self.xmin:.1f} and {self.xmax:.1f} cm-1"
                             f" / {naj:.2f}\n")
        self.fig.canvas.draw_idle()

    def scroll_spectra(self, val):
        """Use the slider to scroll through individual spectra"""
        frame = int(self.sframe.val)
        current_spectrum = self.da.data[frame]
        self.spectrumplot.set_ydata(current_spectrum)
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.titled(self.axspectrum, frame)
        self.fig.canvas.draw_idle()

    def titled(self, ax, frame):
        """Set the title for the spectrum plot"""
        if self.scan_type == "Single":
            new_title = self.da.attrs["Title"]
        elif self.scan_type == "Series":
            new_title = f"Spectrum @ {np.datetime64(self.da.Time.data[frame], 's')}"
        else:
            try:
                new_title = "Spectrum @ "+\
                    f"{self.da.RowCoord}: {self.da[self.da.RowCoord].data[frame]}"\
                  + f"; {self.da.ColCoord}: {self.da[self.da.ColCoord].data[frame]}"
            except (AttributeError, KeyError):
                new_title = f"S{frame//400:3d} {frame%400:3d}"
        ax.set_title(new_title, x=0.28)
