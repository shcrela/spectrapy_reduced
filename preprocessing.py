#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:53:55 2021

@author: dejan
"""
import inspect
from warnings import warn
import numpy as np
import xarray as xr
from skimage import morphology, filters
from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing, impute, decomposition
import calculate as cc
import visualize as vis
import matplotlib.pyplot as plt
# try:
#     from sklearnex import patch_sklearn
#     patch_sklearn()
# except ModuleNotFoundError:
#     pass


def map2series(da: xr.DataArray, axis: str = "shorter", method: str = "numpy",
               func: str = "median"):
    """Transform a map scan into a series.

    Takes the mean value along the given axis.
    The new `MeasurementType` attribute will become `"Series"`

    Parameters:
    -----------
    da: xr.DataArray
        your datarray containing spectra as constructed with read_WDF
    axis: str
        The axis along which you want to reduce the scan.
        (Values along the other axis will be replaced with a single value
        calculated using `func`.)
        Must be one of ["shorter", "longer", `da.RowCoord`, `da.ColCoord`].
        Default is "shorter".
    method: str
        one of ["numpy", "xarray"]
        `"numpy"` keeps the first instance of `Time` in the agglomerating group,
        which is better, since it can work with timedate values not only timedelta,
        whereas `"xarray"` takes the same (median or mean) for Time
        as for the other coordinates.
        Also, `"xarray"` method is approximatly 20% faster.
    func: str
        The function to apply over the `axis`. Not yet implemented!!!
        For the moment we always calculate the median value.
        (advantages over the mean include the  robustness in the presence
        of cosmic rays and other abberations)
    Returns:
    --------
    The updated input object with initial values along the given axis
    replaced with median values along that axis.
    """

    def m2s(da, gc):
        da.name = "Intensities"
        newda = da.reset_coords(set(da.coords.keys()) - {gc}).groupby(gc).median()
        newda = newda.set_coords(list(da.coords.keys()))
        newda = newda.to_array()
        newda = newda.squeeze("variable", drop=True)
        newda = newda.swap_dims({str(gc): "points"})
        newda.attrs = da.attrs
        return newda

    if axis == "shorter":
        ax = np.argmax(da.ScanShape)
        gc = [da.RowCoord, da.ColCoord][ax]
    elif axis == "longer":
        ax = np.argmin(da.ScanShape)
        gc = [da.RowCoord, da.ColCoord][ax]
    elif axis in [da.RowCoord, da.ColCoord]:
        gc = axis
        ax = [da.RowCoord, da.ColCoord].index(axis)
    else:
        raise ValueError('The `axis` argument must be one of ' +
                         f'["shorter", "longer", {da.RowCoord}, {da.ColCoord}].')
    if method == "xarray":
        return m2s(da, gc)
    # Let's create a dictionary to store the recalculated coordinate values
    c_dict = dict()
    for c in da.coords:
        c_dict[c] = []  # a list for each coordinate so that we can append 1 by 1

    for val in np.unique(da[gc].data):  # For each unique value of gc
        # We create a temporary DataArray
        temp_da = da[da[gc] == val]
        for c in da.coords:
            if c == "Time":  # For Time coord, we take the first value
                c_dict[c].append(temp_da[c].data[0])
            else:  # For other coords, we calculate the median
                c_dict[c].append(np.median(temp_da[c].data))

    coord_dict = dict()
    for k, v in c_dict.items():
        coord_dict[k] = ("points", v, da[k].attrs)
    dr = xr.DataArray(data=da.groupby(gc).median().data,
                      dims=("points", "RamanShifts"),
                      coords={**coord_dict,
                              "shifts": ("RamanShifts", da.shifts.data,
                                         {"units": "1/cm"})},
                      attrs=da.attrs)
    dr.attrs["ScanShape"] = (da.ScanShape[ax], 1)
    dr.attrs["Count"] = len(dr.values)
    dr.attrs["MeasurementType"] = "Series"
    dr.attrs["Title"] = f"{da.Title} {da.ScanShape} map turned into" +\
                        f" a {dr.Count} length Series."
    return dr


def gimme_spectra(input_object):
    """Retreive the spectra and ndims"""

    if isinstance(input_object, xr.DataArray):
        spectra = input_object.data
    elif isinstance(input_object, xr.Dataset):
        spectra = input_object.Measured.data
    else:
        spectra = input_object
    return spectra, spectra.ndim


def giveback_same(input_object, output_spectra):

    if isinstance(input_object, xr.DataArray):
        output_object = xr.DataArray(output_spectra,
                                     dims=input_object.dims,
                                     coords=input_object.coords,
                                     attrs=input_object.attrs)
    elif isinstance(input_object, xr.Dataset):
        input_object.Measured.data = output_spectra
        f_name = inspect.stack()[1].function

        return input_object.rename_vars({"Measured": f_name})

    else:
        output_object = output_spectra
    return output_object


def line_up_spectra(spectra):
    old_shape = spectra.shape
    return spectra.reshape(-1, old_shape[-1]), old_shape


def pca_clean(da, n_components='mle', array="Measured", assign=False,
              visualize_clean=False, visualize_components=False,
              visualize_err=False, **kwargs):
    """Clean (smooth) the spectra using PCA.

    Parameters:
    -----------
    da: xr.DataArray or a 3D np.ndarray of shape=(ny,nx,shifts)
        The object containing your input spectra
    n_components: "mle", float, int or None
        "mle":
            The number of components is determined by "mle" algorithm.
        float:
            The variance rate covered.
            The number of components is chosent so to cover the variance rate.
        int:
            The number of components to use for pca decomposition
        None:
            Choses the minimum between the n_features and n_samples
        see more in scikit-learn docs for PCA
    visualize_clean: bool
        Wheather to visualize the result of cleaning
    visualize_components: bool
        Wheather to visualize the decomposition and the components
    assign: bool
        Wheather to assign the results of pca decomposition to the returned
        xr.DataArray.
        If True, the resulting xr.DataArray will be attributed new coordinates:
            `pca_comp[N]` along the `"RamanShifts"` dimension, and
            `pca_comp[N]_coeffs` along the `"points"` dimension
    Returns:
    --------
        updated object with cleaned spectra as .spectra
        spectra_reduced: numpy array
            it is the the attribute added to the WDF object
    """

    spectra = da.data
    shape = da.attrs["ScanShape"] + (-1, )

    pca = decomposition.PCA(n_components, **kwargs)
    pca_fit = pca.fit(spectra)
    spectra_reduced = pca_fit.transform(spectra)
    spectra_cleaned = pca_fit.inverse_transform(spectra_reduced)
    n_components = int(pca_fit.n_components_)
    if visualize_components:
        visualize_components = vis.AllMaps(spectra_reduced.reshape(shape),
                                           components=pca_fit.components_,
                                           components_sigma=da.shifts.data)
        da.attrs["PCA_Components_visu"] = visualize_components

    if visualize_err:
        plt.figure()
        sqerr = np.sum((spectra - spectra_cleaned)**2, axis=-1)
        plt.imshow(sqerr.reshape(da.ScanShape))

    if visualize_clean:
        _s = np.stack((spectra, spectra_cleaned), axis=-1)
        label = ["original spectra", "pca cleaned"]
        visualize_result = vis.ShowSpectra(_s, da.shifts.data,
                                           label=label)
        da.attrs["PCA_Clean_visu"] = visualize_result

    if assign:
        da = da.expand_dims({"components_pca": n_components}, axis=1)
        da = da.assign_coords(pca_components=(("components_pca", "RamanShifts"),
                              pca_fit.components_))
        da = da.assign_coords(pca_mixture_coeffs=(("points", "components_pca"),
                              spectra_reduced))
        return da
    else:
        return giveback_same(da, spectra_cleaned), pca_fit.components_, spectra_reduced


def select_zone(da, on_map=False, **kwargs):
    """Isolate the zone of interest in the input spectra.

    Parameters:
    -----------
        input_spectra: xr.DataArray or np.ndarray of spectra
        x_values: if input_spectra is not xarray, this shoud be given,
                otherwise, a simple np.arrange is used
        left, right : float
            The start and the end of the zone of interest in x_values
            (Ramans shifts)
        if on_map == True:
            left, right, top, bottom: int
                Now those keywords correspond to the zone to be selected
                on the map!
                 left: "from" column, right: "to" column,
                 top: "from" row, bottom: "to" row
    Returns:
    --------
        spectra: the same type of object as `input_spectra`
            Updated object, without anything outside of the zone of interest."""
    input_spectra = da.copy()
    spectra, nedim = gimme_spectra(input_spectra)
    n_y, n_x = input_spectra.ScanShape
    if isinstance(input_spectra, xr.DataArray):
        x_values = input_spectra.shifts.data
    else:
        x_values = kwargs.get("x_values", np.arange(spectra.shape[-1]))
    if on_map is False:
        left = kwargs.get('left', x_values.min())
        right = kwargs.get('right', x_values.max())
        left_ind = np.argmax(x_values >= left)
        right_ind = np.where(x_values <= right)[0][-1]
        if isinstance(input_spectra, xr.DataArray):
            input_spectra.attrs["PointsPerSpectrum"] = right_ind - left_ind
            return input_spectra.sel({"RamanShifts": slice(left_ind, right_ind)})
        else:
            condition = (x_values >= left) & (x_values <= right)
            x_values = x_values[condition]
            spectra = spectra[..., condition]
    elif on_map is True:
        left = kwargs.get('left',
                          input_spectra[input_spectra.ColCoord].data.min())
        right = kwargs.get('right',
                           input_spectra[input_spectra.ColCoord].data.max())
        top = kwargs.get('top',
                         input_spectra[input_spectra.RowCoord].data.min())
        bottom = kwargs.get('bottom',
                            input_spectra[input_spectra.RowCoord].data.max())
        left_ind = np.argmax(np.unique(
            input_spectra[input_spectra.ColCoord].data) > left) - 1
        right_ind = np.argmin(np.unique(
            input_spectra[input_spectra.ColCoord].data) < right)
        top_ind = np.argmax(np.unique(
            input_spectra[input_spectra.RowCoord].data) > top) - 1
        bottom_ind = np.argmin(np.unique(
            input_spectra[input_spectra.RowCoord].data) < bottom)
        assert((top <= bottom) and (left <= right)), \
            "'top' must be <= 'bottom', and 'right <= 'left'!"
        if isinstance(input_spectra, xr.DataArray):
            new_n_y = bottom_ind - top_ind
            new_n_x = right_ind - left_ind
            indices = np.arange(input_spectra.attrs["Count"], dtype=int
                                ).reshape(n_y, n_x)[top_ind:bottom_ind,
                                                    left_ind:right_ind].ravel()
            input_spectra.attrs['NbSteps'][input_spectra.attrs['NbSteps'] > 1]\
                = new_n_x, new_n_y
            input_spectra.attrs['Count'] = new_n_x * new_n_y
            input_spectra.attrs["ScanShape"] = new_n_y, new_n_x
            return input_spectra.sel({"points": indices})

    return giveback_same(input_spectra, spectra)


def normalize(inputspectra, method="robust_scale", **kwargs):
    """
    scale the spectra

    Parameters
    ----------
    inputspectra : xr.DataArray or ndarray
    x_values: should be set if inputspectra is just an array
    method: str
        one of ["l1", "l2", "max", "min_max", "wave_number",
                "robust_scale", "area"]
        default is "robust_scale"
    if method == "robust_scale": the scaling with respect to the
    given quantile range
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html
        quantile : tuple
            default = (5, 95)
        centering: bool
            default = False
    if method == "wave_number":
        wave_number: float
        Sets the intensity at the given wavenumber as 1 an the rest is scaled
        accordingly.

    Returns
    -------
    xr.DataArray with scaled spectra.

    """
    if isinstance(inputspectra, xr.DataArray):
        spectra = inputspectra.data
        x_values = inputspectra.shifts.data
    else:
        x_values = kwargs.get("x_values", np.arange(inputspectra.shape[-1]))
        spectra = inputspectra

    if method in ["l1", "l2", "max"]:
        normalized_spectra = preprocessing.normalize(spectra, axis=1,
                                                     norm=method, copy=False)
    elif method == "min_max":
        normalized_spectra = preprocessing.minmax_scale(spectra, axis=-1,
                                                        copy=False)
    elif method == "area":
        normalized_spectra = spectra / np.expand_dims(np.trapz(spectra,
                                                      x_values), -1)
    elif method == "wave_number":
        wave_number = kwargs.get("wave_number", x_values.min())
        idx = np.nanargmin(np.abs(x_values - wave_number))
        normalized_spectra = spectra / spectra[..., idx][:, np.newaxis]
    elif method == "robust_scale":
        quantile = kwargs.get("quantile", (5, 95))
        centering = kwargs.get("centering", False)
        normalized_spectra = preprocessing.robust_scale(spectra, axis=-1,
                                                        with_centering=centering,
                                                        quantile_range=quantile)
    else:
        warn('"method" must be one of '
             '["l1", "l2", "max", "min_max", "wave_number", "robust_scale", "area"]')
    normalized_spectra -= np.min(normalized_spectra, axis=-1, keepdims=True)
    return giveback_same(inputspectra, normalized_spectra)


def remove_CRs(inputspectra, nx=0, ny=0, sensitivity=0.01, width=0.02,
               visualize=False, **initialization):
    """Remove the spikes using the similarity of neighbouring spectra.
    ATTENTION: Returns normalized spectra.

    Parameters:
    -----------
        inputspectra: xr.DataArray
            your input spectra
        nx, ny : int
            The number of columns / rows in the map spectra
        sensitivity: float from 0 to 1
            Adjusts the sensitivity (high sensitivity detects weaker spikes)
        width: float from 0 to 1
            How wide you expect your spikes to be

    Returns:
    --------
        outputspectra: xr.DataArray
            Normalized input spectra with (hopefully) no spikes.

    """

    if isinstance(inputspectra, xr.DataArray):
        mock_sp3 = inputspectra.data
        ny, nx = inputspectra.attrs["ScanShape"]
    else:
        mock_sp3 = inputspectra
    # NORMALIZATION:
    mock_sp3 -= np.min(mock_sp3, axis=-1, keepdims=True)
    area = np.median(mock_sp3, axis=-1)[:, np.newaxis]
    mock_sp3 /= area
    # Define the neighbourhood:
    neighbourhood = morphology.disk(5)[:, :, None]
    # construct array so that each pixel has the median value of the hood:
    median_spectra3 = filters.median(mock_sp3.reshape(ny, nx, -1),
                                     footprint=neighbourhood).reshape(ny*nx, -1)
    # I will only take into account the positive values (CR):
    coarsing_diff = (mock_sp3 - median_spectra3)
    # To find the bad neighbours :
    bad_neighbours = np.nonzero(coarsing_diff > 10)
    # (1/sensitivity)*\
    #                                              np.std(coarsing_diff))

    if len(bad_neighbours[0]) == 0:
        v = None
        print("No Cosmic Rays found!")
    else:
        print(len(bad_neighbours[0]))
        # =====================================================================
        #             We want to extend the "bad neighbour" label
        #               to adjecent points in each such spectra:
        # =====================================================================
        mask = np.zeros_like(mock_sp3)
        mask[bad_neighbours] = 1
        wn_window_size = int(width*mock_sp3.shape[-1])
        window = np.ones((1, wn_window_size))
        mask = morphology.binary_dilation(mask, footprint=window)
        mock_sp3[mask] = median_spectra3[mask]
        if visualize:
            _s = np.stack((inputspectra[bad_neighbours[0]],
                           median_spectra3[bad_neighbours[0]]), axis=-1)
            v = vis.ShowSpectra(_s, labels=["original", "corrected"])

    return giveback_same(inputspectra, mock_sp3*area), v
