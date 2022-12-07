#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:13:46 2021

@author: dejan
"""
from os import path
from warnings import warn
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io, draw
from sklearn import decomposition
from matplotlib.widgets import (Slider, Button, RadioButtons, SpanSelector,
                                CheckButtons, MultiCursor)
import calculate as cc
import preprocessing as pp
from read_WDF import get_exif


def show_grid(da, img):
    """Show the image and over it the grid of scan points"""
    if img is None:
        print("No image was recovered!")
        return
    def fun_form(i):
        """Transform pixel coordinates to the real XY coordinates"""
        def coord_pos(x, pos, i=i):
            """Only one function : i=0 for x and i=1 for y"""
            xind = int(x)
            if xind <= int([img.width, img.height][i]):
                xxorig = float(img_exif['FocalPlaneXYOrigins'][i])
                xx = f"{np.round(xxorig + xind/[xres, yres][i])}µm"
            else:
                xx = ""
            return xx
        # returns the function:
        return coord_pos

    img_exif = get_exif(img)
    img_arr = np.array(img.getdata()).reshape(img.height, img.width, 3)

    xres = img.width / float(img_exif["FocalPlaneXResolution"])  # in px/µm
    yres = img.height / float(img_exif["FocalPlaneYResolution"])  # in px/µm

    xminpx = round((da.InitialCoordinates[0] - img_exif["FocalPlaneXYOrigins"][0]) * xres)
    yminpx = round((da.InitialCoordinates[1] - img_exif["FocalPlaneXYOrigins"][1]) * yres)

    xmaxpx = xminpx + round(da.StepSizes[0] * da.NbSteps[0] * xres)
    ymaxpx = yminpx + round(da.StepSizes[1] * da.NbSteps[1] * yres)

    xminpx, xmaxpx = np.sort([xminpx, xmaxpx])
    yminpx, ymaxpx = np.sort([yminpx, ymaxpx])

    xsizepx = xmaxpx - xminpx
    ysizepx = ymaxpx - yminpx

    grid_in_image: bool = (xsizepx <= img.width) & (ysizepx <= img.height)

    x_pxvals = np.linspace(xminpx, xmaxpx, da.NbSteps[0])
    y_pxvals = np.linspace(yminpx, ymaxpx, da.NbSteps[1])
    fig, ax = plt.subplots()
    if grid_in_image:
        ax.imshow(img_arr)

    for xxx in x_pxvals:
        ax.vlines(xxx, ymin=yminpx, ymax=ymaxpx, lw=1, alpha=0.2)
    for yyy in y_pxvals:
        ax.hlines(yyy, xmin=xminpx, xmax=xmaxpx, lw=1, alpha=0.2)
    ax.scatter(xminpx, yminpx, marker="X", s=30, c='r')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fun_form(0)))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fun_form(1)))

    if not grid_in_image:
        ax.imshow(img_arr)

    fig.show()



def set_img_coordinates(da, ax, unit="µm",
                        rowcoord_arr=None, colcoord_arr=None):

    if rowcoord_arr == None:
        rowcoord_arr = np.unique(da[da.RowCoord].data)
    if colcoord_arr == None:
        colcoord_arr = np.unique(da[da.ColCoord].data)

    def row_coord(y, pos):
        yind = int(y)
        if yind < len(rowcoord_arr):
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


class ShowCollection(object):
    """Visualize a collection of images.

    Parameters
    ----------
    image_pattern : str
        Can take asterixes as wildcards. For ex.: "./my_images/*.jpg" to select
        all the .jpg images from the folder "my_images"
    load_func : function
        The function to apply when loading the images
    first_frame : int
        The frame from which you want to stard your slideshow
    load_func_kwargs : dict
        The named arguments of the load function

    Outputs
    -------
    Interactive graph displaying the images one by one, whilst you can
    scroll trough the collection using the slider or the keyboard arrows

    Example
    -------
    >>> import numpy as np
    >>> from skimage import io, transform

    >>> def binarization_load(f, shape=(132,132)):
    >>>     im = io.imread(f, as_gray=True)
    >>>     return transform.resize(im, shape, anti_aliasing=True)

    >>> ss = ShowCollection(images, load_func=binarization_load,
                            shape=(128,128))
    """

    def __init__(self, image_pattern, load_func=io.imread, first_frame=0,
                 **load_func_kwargs):

        self.coll_all = io.ImageCollection(image_pattern, load_func=load_func,
                                           **load_func_kwargs)
        self.first_frame = first_frame
        self.nb_pixels = self.coll_all[0].size
        self.titles = np.arange(len(self.coll_all))
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.2)
        self.last_frame = len(self.coll_all)-1
        self.line = plt.imshow(self.coll_all[self.first_frame])
        self.ax.set_title(f"{self.titles[self.first_frame]}")

        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=self.axcolor)

        self.sframe = Slider(self.axframe, 'Frame', self.first_frame,
                             self.last_frame, valinit=self.first_frame,
                             valfmt='%d', valstep=1)
        # calls the update function when changing the slider position
        self.sframe.on_changed(self.update)

        # Calling the press function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)

        # self.fig.show()

    def update(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.sframe.val)
        img = self.coll_all[frame]
        self.line.set_data(img)
        self.ax.set_title(f"{self.titles[frame]}")
        self.fig.canvas.draw_idle()

    def press(self, event):
        """Use left and right arrow keys to scroll through frames one by one"""
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < len(self.coll_all)-1:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        img = self.coll_all[new_frame]
        self.line.set_data(img)
        self.ax.set_title(f"{self.titles[new_frame]}")
        self.fig.canvas.draw_idle()


# %%

class AllMaps(object):
    """Rapidly visualize maps of Raman spectra.

    You can also choose to visualize the map and plot the
    corresponding component side by side if you set the
    "components" parameter.

    Parameters
    ----------
    input_spectra : xr.DataArray
    components: 2D ndarray
        The most evident use-case would be to help visualize the decomposition
        results from PCA or NMF. In this case, the function will plot the
        component with the corresponding map visualization of the given
        components' presence in each of the points in the map.
        So, in this case, your input_spectra would be for example
        the matrix of components' contributions in each spectrum,
        while the "components" array will be your actual components.
        In this case you can ommit your sigma values or set them to
        something like np.arange(n_components)
    components_sigma: 1D ndarray
        in the case explained above, this would be the actual wavenumbers
    **kwargs: dict
        can only take 'title' as a key for the moment

    Returns
    -------
    The interactive visualization.
    (you can scroll through sigma values with a slider,
     or using left/right keyboard arrows)
    """

    def __init__(self, input_spectra, sigma=None, components=None,
                 components_sigma=None, interpolation="gaussian",
                 percentile_range=[0, 100], **kwargs):

        if isinstance(input_spectra, xr.DataArray):
            shape = input_spectra.attrs["ScanShape"]
            self.map_spectra = input_spectra.data.reshape(shape + (-1,))
            self.sigma = input_spectra.shifts.data
        else:
            self.map_spectra = input_spectra
            self.sigma = sigma
            if sigma is None:
                self.sigma = np.arange(self.map_spectra.shape[-1])
        assert self.map_spectra.shape[-1] == len(
                self.sigma), "Check your Ramans shifts array"
        self.percentile_range = percentile_range
        self.first_frame = 0
        self.last_frame = len(self.sigma)-1

        self.components = components
        if self.components is not None:
            if components_sigma is None:
                self.components_sigma = np.arange(components.shape[-1])
            else:
                self.components_sigma = components_sigma
            self.fig, (self.ax2, self.ax, self.cbax) = plt.subplots(
                ncols=3, gridspec_kw={'width_ratios': [40, 40, 1]})
            self.cbax.set_box_aspect(
                40*self.map_spectra.shape[0]/self.map_spectra.shape[1])
        else:
            self.fig, (self.ax, self.cbax) = plt.subplots(
                ncols=2, gridspec_kw={'width_ratios': [40, 1]})
            self.cbax.set_box_aspect(
                40*self.map_spectra.shape[0]/self.map_spectra.shape[1])
            # self.cbax = self.fig.add_axes([0.92, 0.3, 0.03, 0.48])
        # Create some space for the slider:
        self.fig.subplots_adjust(bottom=0.19, right=0.89)
        self.title = kwargs.get('title', None)

        self.im = self.ax.imshow(self.map_spectra[:, :, 0],
                                 interpolation=interpolation)
        self.im.set_clim(np.percentile(self.map_spectra[:, :, 0], [1, 99]))
        if self.components is not None:
            self.line, = self.ax2.plot(
                self.components_sigma, self.components[0])
            self.ax2.set_box_aspect(
                self.map_spectra.shape[0]/self.map_spectra.shape[1])
            self.ax2.set_title(f"Component {0}")
        self.titled(0)
        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = self.fig.add_axes(
            [0.15, 0.1, 0.7, 0.03], facecolor=self.axcolor)

        self.sframe = Slider(self.axframe, 'Frame',
                             self.first_frame, self.last_frame,
                             valinit=self.first_frame, valfmt='%d', valstep=1)

        self.my_cbar = mpl.colorbar.Colorbar(self.cbax, self.im)

        # calls the "update" function when changing the slider position
        self.sframe.on_changed(self.update)
        # Calling the "press" function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        # self.fig.show()

    def titled(self, frame):
        if self.components is None:
            if self.title is None:
                self.ax.set_title(f"Raman shift = {self.sigma[frame]:.1f}cm⁻¹")
            else:
                self.ax.set_title(f"{self.title} n°{frame}")
        else:
            self.ax2.set_title(f"Component {frame}")
            if self.title is None:
                self.ax.set_title(f"Component n°{frame} contribution")
            else:
                self.ax.set_title(f"{self.title} n°{frame}")

    def update(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.sframe.val)
        img = self.map_spectra[:, :, frame]
        self.im.set_data(img)
        self.im.set_clim(np.percentile(img, self.percentile_range))
        if self.components is not None:
            self.line.set_ydata(self.components[frame])
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()

    def press(self, event):
        """Use left and right arrow keys to scroll through frames one by one."""
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < len(self.sigma)-1:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        img = self.map_spectra[:, :, new_frame]
        self.im.set_data(img)
        self.im.set_clim(np.percentile(img, self.percentile_range))
        self.titled(new_frame)
        if self.components is not None:
            self.line.set_ydata(self.components[new_frame])
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.fig.canvas.draw_idle()

# %%


class ShowSpectra(object):
    """Rapidly visualize Raman spectra.

    Imortant: Your spectra can either be a 2D ndarray
    (1st dimension is for counting the spectra,
    the 2nd dimension is for the intensities)
    And that would be the standard use-case, But:
    Your spectra can also be a 3D ndarray,
    In which case the last dimension is used to store additional spectra
    (for the same pixel)
    Fo example, you can store spectra, the baseline
    and the corrected spectra all together.
    Parameters:
    -----------
    input_spectra = xr.DataArray or numpy ndarray
        in the latter case, you can provide multiple spectra stacked along the
        last axis
    sigma:
        what's on the x-axis, optional
    title: str or iterable of the same length as the spectra, optional
    labels: list of labels

    Returns
    -------
    The interactive visualization.\n
    (you can scroll through the spectra with a slider,
     or using left/right keyboard arrows)

    Note:
        When there's only one spectrum to visualize, it bugs.
    """

    def __init__(self, input_spectra, sigma=None, **kwargs):

        if isinstance(input_spectra, xr.DataArray):
            self.my_spectra = input_spectra.data
            self.sigma = input_spectra.shifts.data
        else:
            self.my_spectra = input_spectra
            if sigma is None:
                if self.my_spectra.ndim == 1:
                    self.sigma = np.arange(len(self.my_spectra))
                else:
                    self.sigma = np.arange(self.my_spectra.shape[1])
            else:
                self.sigma = sigma
        if self.my_spectra.ndim == 1:
            self.my_spectra = self.my_spectra[np.newaxis, :, np.newaxis]
        if self.my_spectra.ndim == 2:
            self.my_spectra = self.my_spectra[:, :, np.newaxis]

        assert self.my_spectra.shape[1] == len(self.sigma),\
               "Check your Raman shifts array. The dimensions " + \
               f"of your spectra ({self.my_spectra.shape[1]}) and that of " + \
               f"your Ramans shifts ({len(self.sigma)}) are not the same."

        self.first_frame = 0
        self.last_frame = len(self.my_spectra)-1
        self.fig, self.ax = plt.subplots()
        # Create some space for the slider:
        self.fig.subplots_adjust(bottom=0.19, right=0.89)
        self.title = kwargs.get('title', None)
        self.label = kwargs.get('labels', [None])
        if (not hasattr(self.label[0], '__iter__')\
           or len(self.label[0]) != self.my_spectra.shape[0]\
           or isinstance(self.label[0], str))\
           and self.label[0] is not None:
            self.label = [self.label]*self.my_spectra.shape[0]
            self.spectrumplot = self.ax.plot(self.sigma, self.my_spectra[0],
                                             label=self.label[0])
        else:
            self.spectrumplot = self.ax.plot(self.sigma, self.my_spectra[0])

        self.titled(0)
        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = self.fig.add_axes([0.15, 0.1, 0.7, 0.03])
        # self.axframe.plot(self.sigma, np.median(self.my_spectra, axis=0))
        if len(self.my_spectra) > 1:
            self.sframe = Slider(self.axframe, 'N°',
                                 self.first_frame, self.last_frame, valfmt='%d',
                                 valinit=self.first_frame, valstep=1)
            # calls the "update" function when changing the slider position
            self.sframe.on_changed(self.update)
            # Calling the "press" function on keypress event
            # (only arrow keys left and right work)
            self.fig.canvas.mpl_connect('key_press_event', self.press)
        else:
            self.axframe.axis('off')
        # self.fig.show()

    def titled(self, frame):
        if self.title is None:
            self.ax.set_title(f"Spectrum N° {frame} /{self.last_frame + 1}")
        elif isinstance(self.title, str):
            self.ax.set_title(f"{self.title} n°{frame}")
        elif hasattr(self.title, '__iter__'):
            self.ax.set_title(f"{self.title[frame]}")
        if self.label[0] is not None:
            handles, _ = self.ax.get_legend_handles_labels()
            self.ax.legend(handles, self.label[frame])


    def update(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.sframe.val)
        current_spectrum = self.my_spectra[frame]
        for i, line in enumerate(self.spectrumplot):
            # self.ax.cla()
            line.set_ydata(current_spectrum[:, i])
            self.ax.relim()
            self.ax.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()

    def press(self, event):
        """Use left and right arrow keys to scroll through frames one by one."""
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < self.last_frame:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        current_spectrum = self.my_spectra[new_frame]
        for i, line in enumerate(self.spectrumplot):
            line.set_ydata(current_spectrum[:, i])
            self.ax.relim()
            self.ax.autoscale_view()
        self.titled(new_frame)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

# %%


class NavigationButtons(object):
    """Interactivly visualize multispectral data.

    Navigate trough your spectra by simply clicking on the navigation buttons.

    Parameters
    ----------
        sigma: 1D ndarray
            1D numpy array of your x-values (raman shifts, par ex.)
        spectra: 2D or 3D ndarray
            3D or 2D ndarray of shape (n_spectra, len(sigma), n_curves).
            The last dimension may be ommited it there is only one curve
            to be plotted for each spectra).
        autoscale: bool
            determining if you want to adjust the scale to each spectrum
        title: str
            The initial title describing where the spectra comes from
        label: list
            A list explaining each of the curves. len(label) = n_curves

    Output
    ------
        matplotlib graph with navigation buttons to cycle through spectra

    Example
    -------
        Let's say you have a ndarray containing 10 spectra,
        and let's suppose each of those spectra contains 500 points.

        >>> my_spectra.shape
        (10, 500)
        >>> sigma.shape
        (500, )

        Then let's say you show the results of baseline substraction.

        >>> my_baseline[i] = baseline_function(my_spectra[i])
        your baseline should have the same shape as your initial spectra.
        >>> multiple_curves_to_plot = np.stack(
                (my_spectra, my_baseline, my_spectra - my_baseline), axis=-1)
        >>> NavigationButtons(sigma, multiple_curves_to_plot)
    """
    ind = 0

    def __init__(self, spectra, sigma=None, autoscale_y=False, title='Spectrum',
                 label=False, as_series=False, axis="shorter", **kwargs):

        if as_series:
            spectra = pp.as_series(spectra)
        if isinstance(spectra, xr.DataArray):
            sigma = spectra.shifts.data
            spectra = spectra.data
        elif sigma==None:
            sigma = np.arange(spectra.shape[-1])

        if len(spectra.shape) == 2:
            self.s = spectra[:, :, np.newaxis]
        elif len(spectra.shape) == 3:
            self.s = spectra
        else:
            raise ValueError("Check the shape of your spectra.\n"
                             "It should be (n_spectra, n_points, n_curves)\n"
                             "(this last dimension might be ommited"
                             "if it's equal to one)")
        self.y_autoscale = autoscale_y
        self.n_spectra = self.s.shape[0]
        if isinstance(title, list) or isinstance(title, np.ndarray):
            if len(title) == spectra.shape[0]:
                self.title = title
            else:
                raise ValueError(f"you have {len(title)} titles,\n"
                                 f"but you have {len(spectra)} spectra")
        else:
            self.title = [title]*self.n_spectra

        self.sigma = sigma
        if label:
            if len(label) == self.s.shape[2]:
                self.label = label
            else:
                warn(
                    "You should check the length of your label list.\n"
                    "Falling on to default labels...")
                self.label = ["Curve n°"+str(numb)
                              for numb in range(self.s.shape[2])]
        else:
            self.label = ["Curve n°"+str(numb)
                          for numb in range(self.s.shape[2])]

        self.figr, self.axr = plt.subplots(**kwargs)
        self.axr.set_title(f'{title[0]}')
        self.figr.subplots_adjust(bottom=0.2)
        # l potentially contains multiple lines
        self.line = self.axr.plot(self.sigma, self.s[0], lw=2, alpha=0.7)
        self.axr.legend(self.line, self.label)
        self.axprev1000 = plt.axes([0.097, 0.05, 0.1, 0.04])
        self.axprev100 = plt.axes([0.198, 0.05, 0.1, 0.04])
        self.axprev10 = plt.axes([0.299, 0.05, 0.1, 0.04])
        self.axprev1 = plt.axes([0.4, 0.05, 0.1, 0.04])
        self.axnext1 = plt.axes([0.501, 0.05, 0.1, 0.04])
        self.axnext10 = plt.axes([0.602, 0.05, 0.1, 0.04])
        self.axnext100 = plt.axes([0.703, 0.05, 0.1, 0.04])
        self.axnext1000 = plt.axes([0.804, 0.05, 0.1, 0.04])

        self.bprev1000 = Button(self.axprev1000, 'Prev.1000')
        self.bprev1000.on_clicked(self.prev1000)
        self.bprev100 = Button(self.axprev100, 'Prev.100')
        self.bprev100.on_clicked(self.prev100)
        self.bprev10 = Button(self.axprev10, 'Prev.10')
        self.bprev10.on_clicked(self.prev10)
        self.bprev = Button(self.axprev1, 'Prev.1')
        self.bprev.on_clicked(self.prev1)
        self.bnext = Button(self.axnext1, 'Next1')
        self.bnext.on_clicked(self.next1)
        self.bnext10 = Button(self.axnext10, 'Next10')
        self.bnext10.on_clicked(self.next10)
        self.bnext100 = Button(self.axnext100, 'Next100')
        self.bnext100.on_clicked(self.next100)
        self.bnext1000 = Button(self.axnext1000, 'Next1000')
        self.bnext1000.on_clicked(self.next1000)

    def update_data(self):
        _i = self.ind % self.n_spectra
        for ll in range(len(self.line)):
            yl = self.s[_i][:, ll]
            self.line[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next1(self, event):
        self.ind += 1
        self.update_data()

    def next10(self, event):
        self.ind += 10
        self.update_data()

    def next100(self, event):
        self.ind += 100
        self.update_data()

    def next1000(self, event):
        self.ind += 1000
        self.update_data()

    def prev1(self, event):
        self.ind -= 1
        self.update_data()

    def prev10(self, event):
        self.ind -= 10
        self.update_data()

    def prev100(self, event):
        self.ind -= 100
        self.update_data()

    def prev1000(self, event):
        self.ind -= 1000
        self.update_data()

# %%

class ShowSelected(object):
    """To be used for visual exploration of the maps.

    Select a span on the lower plot and a map of a chosen function
    will appear on the image axis.
    Click on the image to see the spectrum corresponding to that pixel
    on the bottom plot.
    Select `Draw profile` and then click and drag to draw line on the image.
    After the line is drawn, you'll see the profile of the value shown on the
    image appear on the lower plot.

    You can use your mouse to select a zone in the spectra and a map plot
    should appear in the upper part of the figure.
    On the left part of the figure you can select what kind of function
    you want to apply on the selected span.
    ['area', 'barycenter_x', 'max_value', 'peak_position', 'peak_ratio']

    Parameters:
    -----------
        da: xarray.DataArray object
            dataArray containing your spectral recordings and some metadata.
        interpolation: string
            indicates to matplotlib what interpolation to use between pixels.
            Must be one of:
            [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
             'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
             'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
            see: https://ouvaton.link/GP0Jti
            https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
        cmap: string
            matplotlib cmap to use. Default is "viridis".
        norm: matplotlib normalization
            Default is `mpl.colors.Normalize(vmin=0, vmax=1)`
            You can use for example `mpl.colors.CenteredNorm(vcenter=0.5))`
            https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
        facecolor: string
            Any acceptable matplotlib color. Default is "`oldlace`".
        active_color: string
            Any acceptable matplotlib color. Default is `'lightcoral'`
        extent: string
            what extent you want matplotlib to use
        figsize: (int, int)
        kwargs: dict
            kwargs to pass on to plt.figure
    """

    def __init__(self, input_spectra, x=None, interpolation='kaiser',
                 epsilon=0.5, **kwargs):

        self.da = input_spectra.copy()
        self.interp = interpolation
        self.f_names = ['area',
                        'barycenter_x',
                        'max_value',
                        'peak_position',
                        'peak_ratio']
        self.shifts = self.da.shifts.data
        self.xmin = self.shifts.min()
        self.xmax = self.shifts.max()
        # Get some basic info on the spectra:
        self.nshifts = self.da.attrs["PointsPerSpectrum"]
        self.ny, self.nx = self.da.attrs["ScanShape"]
        self.scan_type = self.da.attrs["MeasurementType"]
        self.cmap = kwargs.pop("cmap", "cividis")
        self.facecolor = kwargs.pop("facecolor", "oldlace")
        self.active_color = kwargs.pop("active_color", "lightcoral")
        self.file_location = kwargs.pop("file_location",
                                        self.da.attrs["Folder name"])
        self.extent = kwargs.pop("extent", "full")
        self.norm = kwargs.pop("norm", mpl.colors.Normalize(vmin=0, vmax=1))
#                                       mpl.colors.CenteredNorm(vcenter=0.5))
#       #---------------------------- about labels ---------------------------#
        xlabel = self.da.attrs["ColCoord"]
        ylabel = self.da.attrs["RowCoord"]
        if (self.scan_type == "Map") and (self.da.MapAreaType != "Slice"):
            self.xlabel = f"{xlabel} [{input_spectra[xlabel].units}]"
            self.ylabel = f"{ylabel} [{input_spectra[ylabel].units}]"
        else:  # Not a map scan
            self.xlabel = xlabel
            self.ylabel = ylabel
#        #---------------------------------------------------------------------#
        # Preparing the plots:
        figsize = kwargs.pop("figsize", (14, 8))
        self.fig = plt.figure(figsize=figsize, facecolor=self.facecolor,
                              **kwargs)
        # Add all the axes:
        self.aximg = self.fig.add_axes([.21, .3, .77, .6])  # main frame
        self.axspectrum = self.fig.add_axes([.05, .075, .93, .15],
                                            facecolor=self.facecolor)
        self.axradio = self.fig.add_axes([.05, .3, .1, .6],
                                         facecolor=self.facecolor)
        self.axreduce = self.fig.add_axes([.05, .275, .1, .09],
                                          facecolor=self.facecolor)
        self.axabsscale = self.fig.add_axes([.05, .22, .1, .09],
                                            facecolor=self.facecolor)
        self.axprofile = self.fig.add_axes([.05, .9, .06, .05],
                                           facecolor=self.facecolor)
        # self.axscroll = self.fig.add_axes([.05, .02, .9, .02])
        self.axradio.axis('off')
        self.axreduce.axis('off')
        self.axabsscale.axis('off')
        # self.axprofile.axis('off')
        # self.axscroll.axis('off')
        self.first_frame = 0
#        if self.scan_type != "Single":
#            # Slider to scroll through spectra:
#            self.last_frame = len(self.da.data)-1
#            self.sframe = Slider(self.axscroll, 'S.N°',
#                                 self.first_frame, self.last_frame,
#                                 valinit=self.first_frame, valfmt='%d',
#                                 valstep=1)
#            self.sframe.on_changed(self.scroll_spectra)

        # Show the spectrum:
        self.spectrumplot, = self.axspectrum.plot(self.da.shifts.data,
                                                  self.da.data[self.first_frame])
        self.axspectrum.xaxis.set_major_formatter(
                                mpl.ticker.FuncFormatter(self._add_cm_units))
        self.titled(self.axspectrum, self.first_frame)
        self.vline = None
        # The span selector on the spectrumplot:
        self.span = SpanSelector(self.axspectrum, self.onselect, 'horizontal',
                                 useblit=True, span_stays=True,
                                 rectprops=dict(alpha=0.5,
                                                facecolor=self.active_color))
        # Radio buttons for function selection:
        self.func = "area"  # Default function
        self.func_choice = RadioButtons(self.axradio, self.f_names,
                                        activecolor=self.active_color)
        self.func_choice.on_clicked(self.determine_func)
        # The "reduced" button
        self.reduced_button = CheckButtons(self.axreduce, ["reduced"])
        self.reduced_button.on_clicked(self.is_reduced)
        self.reduced = self.reduced_button.get_status()[0]

        self.abs_scale_button = CheckButtons(self.axabsscale, ["abs. scale"])
        self.abs_scale_button.on_clicked(self.is_absolute_scale)
        self.absolute_scale = self.abs_scale_button.get_status()[0]

        self.draw_profile_button = Button(self.axprofile, "Draw\nProfile",
                                          color=self.facecolor)
        self.draw_profile_button.on_clicked(self.start_drawing)

        self.func = self.func_choice.value_selected

        # Plot the empty "image":
        if self.scan_type == "Map":
            self.imup = self.aximg.imshow(cc.calculate_ss(
                                        self.func,
                                        self.da),
                                        interpolation=self.interp,
                                        aspect=self.nx/self.ny/1.4,
                                        cmap=self.cmap,
                                        norm=self.norm,
                                        # vmin=0,
                                        # vmax=1
                                        )
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
                                            bbox={'facecolor': self.active_color,
                                            'alpha': 0.3, 'pad': 10})
        else:
            _length = np.max((self.ny, self.nx))
            self.imup, = self.aximg.plot(np.arange(_length),
                                         np.zeros(_length), '--o', alpha=.5)
#        #--------------For drawing the line----------------------
        self.drawing_enabled = False
        self.button_released = False
        if isinstance(self.imup, mpl.image.AxesImage):  # if image
            self.axes = self.aximg
            self.epsilon = epsilon
            self.fixed_ind = 0
            self.moving_ind = None
            self.line = None
            self.move = False
            self.point = []
            self.line_coords = {0: np.full(2, np.nan),
                                1: np.full(2, np.nan)}
            self.canvas = self.axes.figure.canvas

        self.cidonclick = self.fig.canvas.mpl_connect('button_press_event',
                                                      self.onclick)

    def connect_line_draw(self):
        self.fig.canvas.mpl_disconnect(self.cidonclick)
        self.draw_profile_button.color = self.active_color
        [spine.set_visible(False) for _, spine in self.axprofile.spines.items()]
        self.cidpressline = self.canvas.mpl_connect('button_press_event',
                                                    self.pressline)
        self.cidbuttonrelease = self.canvas.mpl_connect('button_release_event',
                                                        self.button_release_callback)
        self.cidmotionnotify = self.canvas.mpl_connect('motion_notify_event',
                                                       self.motion_notify_callback)

    def disconnect_line_draw(self):
        self.draw_profile_button.color = self.facecolor
        [spine.set_visible(True) for _, spine in self.axprofile.spines.items()]
        self.remove_line()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_disconnect(self.cidbuttonrelease)
        self.canvas.mpl_disconnect(self.cidmotionnotify)
        self.canvas.mpl_disconnect(self.cidpressline)
#        #---------------------------------------------------------------------

    def draw_first_point(self, x0, y0):
        self.remove_line()
        self.point.append(self.axes.plot(x0, y0, 'sr', ms=5, alpha=0.4)[0])
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.axes.bbox)
        self.line, = self.axes.plot(x0, y0, "-b", lw=5, alpha=.6)

    def remove_line(self):
        if isinstance(self.imup, mpl.image.AxesImage):  # if image
            self.point = []
            self.line = None
            while len(self.axes.lines) > 0:
                self.axes.lines.pop()
            self.canvas.draw()

    def pressline(self, event):
        if event.inaxes == self.axes and isinstance(self.imup, mpl.image.AxesImage):
            self.background = self.canvas.copy_from_bbox(self.axes.bbox)
            x1 = event.xdata
            y1 = event.ydata
            self.button_released = False
            self.move = True
            if len(self.point) == 2:  # If there are already two points
                assert self.line is not None, "2 points and no line?"
                xs, ys = self.line.get_data()
                d = np.sqrt((xs - x1)**2 + (ys - y1)**2)
                if min(d) > self.epsilon:  # Draw a new point
                    self.line_coords[0] = np.array((x1, y1))
                    self.fixed_ind = 0
                    self.draw_first_point(*self.line_coords[0])
                else:  # move the existing point and with it, the line
                    self.moving_ind = np.argmin(d)  # What point to move
                    self.fixed_ind = np.argmax(d)
                    self.line_coords[self.fixed_ind] = np.array((xs[self.fixed_ind],
                                                                 ys[self.fixed_ind]))
                    self.line_coords[self.moving_ind] = np.array((xs[self.moving_ind],
                                                                  ys[self.moving_ind]))
                    self.line.set_animated(True)
            else:  # No line present -> Draw the first point
                self.line_coords[0] = np.array((x1, y1))
                self.draw_first_point(*self.line_coords[0])

    def motion_notify_callback(self, event):  # (self, event):

        if (event.inaxes != self.axes) or (self.button_released) or (not self.move):
            return
        xmove = event.xdata
        ymove = event.ydata
        self.line.set_data([self.line_coords[self.fixed_ind][0], xmove],
                           [self.line_coords[self.fixed_ind][1], ymove])
        self.canvas.restore_region(self.background)
        self.axes.draw_artist(self.line)
        self.canvas.blit(self.axes.bbox)

    def button_release_callback(self, event):

        if (event.inaxes == self.axes) and \
                isinstance(self.imup, mpl.image.AxesImage):  # if image
            self.button_released = True
            self.move = False
            newx = event.xdata
            newy = event.ydata
            if len(self.point) == 2:  # This means we just moved en existing line
                self.line_coords[self.moving_ind] = np.array((newx, newy))
                self.line.set_animated(False)
                self.background = None
                for i, p in enumerate(self.point):
                    p.set_data(self.line_coords[i])
                self.canvas.draw()
            else:  # This adds a second point and draws a new line (if dragged).
                self.line_coords[1] = np.array((newx, newy))
                # Check if we moved between clicking and releasing the button:
                if np.sqrt(np.sum((self.line_coords[1] -
                                   self.line_coords[0])**2)) > self.epsilon:
                    self.point.append(self.axes.plot(*self.line_coords[1],
                                                     "sr", ms=5, alpha=.4)[0])
                    self.canvas.draw()
                else:  # This would ammount to clicking without dragging:
                    self.onclick(event)  # Show spectrum of the pixel clicked on
                    return  # Dont go to the next line
            self.draw_pixel_values()
        else:
            return

    def draw_pixel_values(self):
        line_ends = np.round(np.array((*self.line_coords[0],
                                       *self.line_coords[1])).astype(int))

        cc, rr = draw.line(*line_ends)
        my_img = self.imup.get_array()[rr, cc]
        xs = self.da[self.da.ColCoord].data[cc]
        ys = self.da[self.da.RowCoord].data[rr*self.da.ScanShape[1]]
        line_lengths = np.sqrt((xs - xs[0])**2 + (ys - ys[0])**2)
        self.spectrumplot.set_xdata(line_lengths)
        self.axspectrum.xaxis.set_major_formatter(
                            mpl.ticker.FuncFormatter(self._add_micron_units))
        self.spectrumplot.set_ydata(my_img)
        self.axspectrum.set_xlim(0, line_lengths.max())
        self.axspectrum.set_ylim(my_img.min(), my_img.max()*1.05)
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.axspectrum.set_title(f"Values on the profile {line_ends}",
                                  x=0.28, y=-0.45)
        self.axspectrum.figure.canvas.draw_idle()

    def onclick(self, event):
        """Clicking on a pixel will show the spectrum
        corresponding to that pixel on the bottom plot."""
        if event.inaxes == self.aximg and (event.button in [1, 2, 3]):
            x_pos = round(event.xdata)
            y_pos = round(event.ydata)
            if isinstance(self.imup, mpl.image.AxesImage):
                if x_pos <= self.nx and y_pos <= self.ny and x_pos * y_pos >= 0:
                    broj = round(y_pos * self.nx + x_pos)
                    # self.sframe.set_val(broj)
                    self.scroll_spectra(broj)
            elif isinstance(self.imup, mpl.lines.Line2D):
                self.scroll_spectra(x_pos)
            else:
                pass

    def is_reduced(self, label):
        self.reduced = self.reduced_button.get_status()[0]
        self.draw_img()

    def is_absolute_scale(self, label):
        self.absolute_scale = self.abs_scale_button.get_status()[0]
        self.draw_img()

    def _add_micron_units(self, x, pos):
        return f"{x}µm"

    def _add_cm_units(self, x, pos):
        return f"{int(x)}cm-1 "

    def start_drawing(self, event):
        """Turns on and of the possiblity to draw a line on the image
        and show the corresponding profile"""
        self.draw_profile_button.hovercolor = "lightgray"
        if not self.drawing_enabled:
            self.connect_line_draw()
        else:
            self.disconnect_line_draw()
        self.drawing_enabled = not self.drawing_enabled
        self.axprofile.figure.canvas.draw()

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

    def normalize_data(self, mydata):
        return (mydata - np.min(mydata)) / np.ptp(mydata)

    def draw_img(self):
        """Draw/update the image."""
        # calculate the function:
        img = cc.calculate_ss(self.func, self.da, self.xmin, self.xmax,
                              is_reduced=self.reduced)
        # img = self.normalize_data(img)
        if self.scan_type == "Map":
            if len(self.point) == 2:
                self.draw_pixel_values()
            limits = np.percentile(img, [0, 99])
#            limits = np.ptp(img)
            if self.absolute_scale:
                img = self.normalize_data(img)
                limits = [0, 1]
            self.imup.set_clim(limits)
            self.cbar.mappable.set_clim(*limits)
            self.imup.set_data(img)
        elif self.scan_type == 'Single':
            self.imup.set_text(f"{img[0][0]:.3G}")
        else:
            self.imup.set_ydata(img.squeeze())
            self.aximg.relim()
            self.aximg.autoscale_view(None, False, True)

        self.aximg.set_title(f"Calculated {'reduced'*self.reduced} {self.func} "
                             # f"between {self.xmin:.1f} and {self.xmax:.1f} cm-1"
                             # f" / {naj:.2f}\n"
                             )
        self.fig.canvas.draw_idle()

    def scroll_spectra(self, val):
        """Update the spectrum plot"""
        frame = val  # int(self.sframe.val)
        current_spectrum = self.da.data[frame]
        self.spectrumplot.set_xdata(self.shifts)
        self.spectrumplot.set_ydata(current_spectrum)
        self.axspectrum.set_xlim(self.shifts.min(), self.shifts.max())
        self.axspectrum.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(self._add_cm_units))
        self.axspectrum.set_ylim(current_spectrum.min(),
                                 current_spectrum.max()*1.05)
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.titled(self.axspectrum, frame)
        self.fig.canvas.draw_idle()

    def titled(self, ax, frame):
        """Set the title for the spectrum plot"""
        if self.scan_type == "Single":
            new_title = self.da.attrs["Title"]
        elif self.scan_type == "Series":
            if "Temperature" in self.da.coords.keys():
                coordinate = f"{self.da.Temperature.data[frame]} °C"
            else:
                coordinate = f"{np.datetime64(self.da.Time.data[frame], 's')}"
            new_title = f"Spectrum @ {coordinate}"
        else:
            try:
                new_title = f"Spectrum N°{int(frame)} @ " +\
                    f"{self.da.RowCoord}: {self.da[self.da.RowCoord].data[frame]}µm"\
                  + f"; {self.da.ColCoord}: {self.da[self.da.ColCoord].data[frame]}µm"
            except (AttributeError, KeyError):
                # This needs to be made more generic:
                new_title = f"S{frame//self.nx:3d} {frame%self.nx:3d}"
        ax.set_title(new_title, x=0.28, y=-0.45)

    def set_facecolor(self, my_col):
        self.fig.set_facecolor(my_col)
        self.axspectrum.set_facecolor(my_col)
        self.axprofile.set_facecolor(my_col)
        self.axradio.set_facecolor(my_col)
#        self.axradio.update({"facecolor": my_col})
#        self.axradio.figure.canvas.draw()
        self.axreduce.set_facecolor(my_col)
        self.axabsscale.set_facecolor(my_col)
        self.draw_profile_button.color = my_col
        self.facecolor = my_col




class FindBaseline(object):
    """Visualy adjust parameters for the baseline.

    Parameters
    ----------
    my_spectra: 2D ndarray

    Returns
    -------
    The interactive graph facilitating the parameter search.
    You can later recover the parameters with:
        MyFindBaselineInstance.p_val
        MyFindBaselineInstance.lam_val

    Note that you can use the same function for smoothing
    (by setting the `p_val` to 0.5 and `lam_val` to some "small" value (like 13))
    """

    def __init__(self, my_spectra, sigma=None, **kwargs):
        if my_spectra.ndim == 1:
            self.my_spectra = my_spectra[np.newaxis, :]
        else:
            self.my_spectra = my_spectra
        if sigma is None:
            self.sigma = np.arange(my_spectra.shape[1])
        else:
            assert my_spectra.shape[-1] == len(
                sigma), "Check your Raman shifts array"
            self.sigma = sigma

        self.nb_spectra = len(self.my_spectra)
        self.current_spectrum = self.my_spectra[0]
        self.title = kwargs.get('title', None)
        self.p_val = 5e-5
        self.lam_val = 1e5

        self.fig = plt.figure(figsize=(14, 10))
        # Add all the axes:
        self.ax = self.fig.add_axes([.2, .15, .75, .8])  # [left, bottom, width, height]
        self.axpslider = self.fig.add_axes([.05, .15, .02, .8], yscale='log')
        self.axlamslider = self.fig.add_axes([.1, .15, .02, .8], yscale='log')
        if self.nb_spectra > 1:  # scroll through spectra if there are many
            self.axspectrumslider = self.fig.add_axes([.2, .05, .75, .02])
            self.spectrumslider = Slider(self.axspectrumslider, 'Frame',
                                         0, self.nb_spectra-1,
                                         valinit=0, valfmt='%d', valstep=1)
            self.spectrumslider.on_changed(self.spectrumupdate)

        self.pslider = Slider(self.axpslider, 'p-value',
                              1e-10, 0.99, valfmt='%.2g',
                              valinit=self.p_val,
                              orientation='vertical')
        self.lamslider = Slider(self.axlamslider, 'lam-value',
                                .1, 1e10, valfmt='%.2g',
                                valinit=self.lam_val,
                                orientation='vertical')
        self.pslider.on_changed(self.blupdate)
        self.lamslider.on_changed(self.blupdate)

        self.spectrumplot, = self.ax.plot(self.sigma, self.current_spectrum,
                                          label="original spectrum")
        self.bl = cc.baseline_als(self.current_spectrum, p=self.p_val,
                                  lam=self.lam_val, niter=41)
        self.blplot, = self.ax.plot(self.sigma, self.bl, label="baseline")
        self.corrplot, = self.ax.plot(self.sigma,
                                      self.current_spectrum - self.bl,
                                      label="corrected_plot")
        self.ax.legend()
        self.titled(0)

        # self.fig.show()

    def titled(self, frame):
        if self.title is None:
            self.ax.set_title(f"Spectrum N° {frame} /{self.nb_spectra}")
        else:
            self.ax.set_title(f"{self.title} n°{frame}")

    def spectrumupdate(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.spectrumslider.val)
        self.current_spectrum = self.my_spectra[frame]
        self.spectrumplot.set_ydata(self.current_spectrum)
        self.blupdate(val)
        self.ax.relim()
        self.ax.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()

    def blupdate(self, val):
        self.p_val = self.pslider.val
        self.lam_val = self.lamslider.val
        self.bl = cc.baseline_als(self.current_spectrum, p=self.p_val,
                                  lam=self.lam_val)
        self.blplot.set_ydata(self.bl)
        self.corrplot.set_ydata(self.current_spectrum - self.bl)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()


class ChooseComponents(object):
    """Choose which components to use to reconstruct the spectra with.
    
    Parameters:
    -----------
    da: xarray.DataArray
        object containing your spectra and some basic metadata
    n_components: int
        the number of components to use in decomposition
    method: 
    """

    def __init__(self, da, n_components, method, **kwargs):
        print(kwargs)
        self.method = method
        self.spectra = da.data
        self.spectra -= np.min(self.spectra, axis=-1, keepdims=True)
        self.original_spectra = np.copy(da.data)
        self.mean_spectra = np.mean(self.spectra, axis=-1, keepdims=True)
        self.mean_spectra2 = np.mean(self.spectra, axis=0, keepdims=False)
        self.x_values = da.shifts.data
        self.shape = da.attrs["ScanShape"]# + (-1, )
        self.n = n = n_components
        self.x_ind = 0
        self.point_ind = 0
        model = self.method(self.n, **kwargs)
        # Seems to produce better interprable results:
        self.model_fit = model.fit(self.spectra.T)
        self.components = self.model_fit.transform(self.spectra.T).T
        self.coeffs = self.model_fit.components_.T

        # Construct the figure:
        self.fig = plt.figure()
        # Add all the axes and the buttons:
        self.aximg = self.fig.add_axes([.31, .3, .64, .6])  # image
        self.img_data = self.aximg.imshow(self.mean_spectra.reshape(self.shape))
        limits = (1, 1 + np.max(np.ptp(self.spectra, axis=0)))
        self.img_data.set_clim(limits)
        self.axspectrum = self.fig.add_axes([.05, .075, .9, .15])  # spectrum
        self.spec_data, = self.axspectrum.plot(self.x_values,
                                               self.mean_spectra2,
                                               'ob', alpha=.2)
        self.orig_spec_data, = self.axspectrum.plot(self.x_values,
                                                   self.mean_spectra2,
                                                   '--k', alpha=.9)
        self.axspectrum.vline_present = False
        # self.axscroll = self.add_naked(self.fig, [.05, .02, .9, .02])  # for the slider?
        # self.slider = Slider(self.axscroll, r'$\lambda$',
        #                      self.x_values.min(), self.x_values.max(),
        #                      valinit = self.x_values.min(), valstep=1)
        # self.slider.on_changed(self.slider_change)
        # for the "all" button:
        self.axall = self.add_naked(self.fig, [.05, .91, .08, .03])
        self.button_all = Button(self.axall, "All")
        # for the "none" button
        self.axnone = self.add_naked(self.fig, [.15, .91, .08, .03])
        self.button_none = Button(self.axnone, "None")
        # for the checkboxes (to select components):
        self.axchck = self.add_naked(self.fig, [0.01, 0.3, 0.04, 0.6])
        self.color_spines(self.axchck, "w")
        self.comp_labels = ["c"+str(i) for i in range(self.n)]
        self.check = CheckButtons(self.axchck, self.comp_labels, [True]*self.n)
        # Add the axes and plot the components, also adjust the checkbuttons:
        self.axcomp = []
        # print(self.components.shape)
        for i, comp in enumerate(self.components):
            self.axcomp.append(self.add_naked(self.fig,
                                              [.05, .9-(i+1)*.6/n, .18, .6/n]))
            self.axcomp[i].plot(self.x_values, comp)
            self.axcomp[i].vline_present = False
            # Now we need to adjust the positions of the checkboxes (rectangles,
            # lines and labels) for them to be aligned with axcomp axes:
            y = 1-(i+.7)/n
            height = .4/n
            self.check.rectangles[i].set(x=.65, y=y, width=.3, height=height)
            self.check.lines[i][0].set_data([.65, .95], [y+height, y])
            self.check.lines[i][1].set_data([.65, .95], [y, y+height])
            self.check.labels[i].set_position((.02, y+height/2))
        # Add the interactivity
        self.button_all.on_clicked(self.select_all)
        self.button_none.on_clicked(self.select_none)
        self.check.on_clicked(self.just_checking)
        self.spec_cursor = MultiCursor(self.fig.canvas, [self.axspectrum],
                                     color='r', lw=1)
        self.pos_cid = self.fig.canvas.mpl_connect('button_press_event',
                                                   self.draw_vline)
        self.img_cid = self.fig.canvas.mpl_connect('button_press_event',
                                                   self.onclick)

    def add_naked(self, fig, pos):
        """Add new "naked" axes to the figure at the given position `pos`.

        Parameters:
        -----------
        pos: list of floats
            [left, bottom, width, height]
        """
        ax = fig.add_axes(pos)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(bottom=False, left=False)
        return ax

    def color_spines(self, ax, color):
        """Change the color of all spines for an axes"""
        for i in ax.spines:
            ax.spines[i].set(edgecolor=color)#, linewidth=3)

    def switch_spine_color(self, ax):
        """Switch spines color from red to green (and vice versa)."""
        for i in ax.spines:
            boja = np.array(ax.spines[i]._edgecolor, dtype=bool)
            nova_boja = boja + np.array([1, 0, 1, 0], dtype=bool)
            ax.spines[i].set(edgecolor=nova_boja)

    def select_all(self, event):
        """Set all checkboxes to active"""
        for i, s in enumerate(self.check.get_status()):
            if not s:
                self.check.set_active(i)
        self.just_checking("all")

    def select_none(self, event):
        """Set all checkboxes to inactive"""
        for i, s in enumerate(self.check.get_status()):
            if s:
                self.check.set_active(i)
        self.just_checking("none")

    def just_checking(self, label):
        facecolors = ["lightgray", "white"]
        checked = self.check.get_status()
        if label in self.comp_labels:
            clicked_idx = self.comp_labels.index(label)
            color_idx = checked[clicked_idx] ^ False
            self.axcomp[clicked_idx].set_facecolor(facecolors[color_idx])
        else:
            for i, c in enumerate(checked):
                self.axcomp[i].set_facecolor(facecolors[c])
        # used_comps = components[checked]
        self.spectra = np.dot(self.coeffs[:, checked],
                              self.components[checked, :])
        if self.method == decomposition.PCA:
            self.spectra += self.mean_spectra
        self.draw_image()
        self.draw_spectrum()

    def draw_image(self):
            new_img_data = self.spectra[:, self.x_ind].reshape(self.shape)
            self.img_data.set_data(new_img_data)
            limits = np.percentile(new_img_data, (1, 99))
            self.img_data.set_clim(limits)
            self.fig.canvas.draw_idle()

    def draw_vline(self, event):
        if event.inaxes in [self.axspectrum]:
            x = event.xdata
            self.x_ind = np.nanargmin(np.abs(self.x_values - x))
        else:
            x = self.x_values[self.x_ind]
        if self.axspectrum.vline_present:
            self.axspectrum.lines.remove(self.axspectrum.vlineid)
        self.axspectrum.vlineid = self.axspectrum.axvline(x, c='g', lw=.5)
        self.axspectrum.vline_present = True
        for i in range(self.n):
            if self.axcomp[i].vline_present:
                self.axcomp[i].lines.remove(self.axcomp[i].vlineid)
            self.axcomp[i].vlineid = self.axcomp[i].axvline(x, c='g', lw=.5)
            self.axcomp[i].vline_present = True
        self.draw_image()

    def onclick(self, event):
        """Right-Clicking on a pixel will show the spectrum
        corresponding to that pixel on the bottom plot"""
        if event.inaxes == self.aximg:
            x_pos = round(event.xdata)
            y_pos = round(event.ydata)
            nx, ny = self.shape[1], self.shape[0]
            if event.button != 1:
                if x_pos <= nx and y_pos <= ny and x_pos * y_pos >= 0:
                    # print(x_pos, y_pos)
                    self.point_ind = round(y_pos * nx + x_pos)
                    self.draw_spectrum()

    # def slider_change(self, val):
    #     self.x_ind = np.nanargmin(np.abs(self.x_values - val))
    #     self.draw_vline(0)

    def draw_spectrum(self):
        self.spec_data.set_ydata(self.spectra[self.point_ind])
        self.orig_spec_data.set_ydata(self.original_spectra[self.point_ind])
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.fig.canvas.draw_idle()


def draw_aggregated(da, style="dark_background", facecolor="black", note=False,
                    binning=np.geomspace, shading="auto", units="cm$^{{-1}}$",
                    cmap="inferno", add=False, **kwargs):
    """Draw aggregated spectra.
    It puts spectra into bins, color intensity reflects the number of spectra
    present in the given bin.

    Parameters:
        da: xr.DataArray
            Your spectra.
        n_bins: int
            The number of bins you want to use
        style:
            one of matplotlib.pyplot.styles.available
        facecolor:
            matplotlib color
        binning:
            The function to use for binning (np.geomspace, np.linspace,...)
        add: bool
            Weather to add the spectra to an existing figure/axes
            or to draw new figure

    """

    def tocm_1(x, pos):
        """Add units to x_values"""
        nonlocal n
        if x < n:
            xx = f"{da.shifts.data[int(x)]:.0f}{units}"
        else:
            xx = ""
        return xx

    def restablish_zero(y, pos):
        """Restablishes zero values (removed because of log)."""
        nonlocal bins
        yind = int(y)
        if yind < len(bins):
            yy = f"{bins[int(y)] - 1:.2g}"
        else:
            yy = ""
        return yy

    def forward(x):
        return np.exp(x)

    def backward(x):
        return np.log(x)

    @mpl.ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{x:.2f}"

    n = len(da.shifts)
    nm = kwargs.pop("n_bins", min(n, len(da.values)))
    da.values -= (np.min(da.values, axis=-1, keepdims=True) - 1) # We can't have zeros
    bins = binning(np.min(da.values), np.max(da.values), nm)
    binda = np.empty((n, nm), dtype=int)
    prd = []
    for i in range(n):
        bin_data = np.digitize(da.values[:, i], bins=bins, right=False)
        prebroj = np.bincount(bin_data, minlength=nm)
        prd.append(len(prebroj))
        binda[i] = prebroj[:nm]

    norm = kwargs.pop("norm", mpl.colors.LogNorm())
    my_title = kwargs.pop("title", "")
    alpha = kwargs.pop("alpha", .6)
    figsize = kwargs.pop("figsize", (18, 10))
    if add:
        fig = plt.gcf()
        ax = plt.gca()
    else:
        fig, ax = plt.subplots(figsize=figsize)
    if facecolor:
        ax.set_facecolor(facecolor)
    with plt.style.context(style):
        plt.pcolormesh(binda.T, norm=norm, shading=shading, cmap=cmap,
                       alpha=alpha)
        plt.title(my_title)
        # poz_y = plt.yticks()[0][1:-1].astype(int)
        # plt.yticks(poz_y, bins[poz_y].astype(int))
        # poz_x = plt.xticks()[0][1:-1].astype(int)
        # plt.xticks(poz_x, da.shifts.data[poz_x].astype(int))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(tocm_1))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(restablish_zero))
        # ax.yaxis.set_major_formatter(major_formatter)
        plt.ylabel("Intensity")
        # scaler = mpl.scale.FuncScale(plt.gca(), (forward, backward))
        # plt.yscale(scaler)
        if note:
            explanation = (f"NOTE: This plot shows all of the {len(da)}"
                           f" spectra binned in {nm} bins.\n"
                           f"The binning is done with {binning.__name__}")
            fig.supxlabel(f"{explanation:<50s}", fontsize=10)#, transform=plt.gca().transAxes)
        try:
            plt.colorbar(shrink=.5, label="Number of spectra in the bin.")
        except:
            pass
    # plt.show();
