# -*- coding: latin-1 -*-
from distutils.log import warn
import os
import io
from datetime import datetime
import numpy as np
import xarray as xr
from PIL import Image, ImageFile
import constants_WDF_class as const

ImageFile.LOAD_TRUNCATED_IMAGES = True


def convert_time(t):
    """Convert the Windows 64bit timestamp to the human-readable format.

    Input:
    -------
        t: timestamp in W64 format (default for .wdf files)
    Output:
    -------
        string formatted to suit local settings
    """
    return datetime.fromtimestamp(t / 1e7 - 11644473600)


def hr_filesize(filesize, suffix="B"):
    for unit in ["", "k", "M", "G", "T", "P", "E", "Z"]:
        if abs(filesize) < 1024.0:
            return f"{filesize:3.1f}{unit}{suffix}"
        filesize /= 1024.0
    return f"{filesize:.1f}Yi{suffix}"


def get_exif(img):
    """Recover exif data from a PIL image"""

    img_exif = dict()
    for tag, value in img._getexif().items():
        decodedTag = const.EXIF_TAGS.get(tag, tag)
        img_exif[decodedTag] = value
    dunit = img_exif["FocalPlaneResolutionUnit"]
    img_exif["FocalPlaneResolutionUnit"] = const.DATA_UNITS.get(dunit, dunit)
    return img_exif


def read_WDF(filename, verbose=False, **kwargs):
    """Read data from the binary .wdf file..

    Example
    -------
    >>> da, img = read_WDF(filename)

    Input
    ------
    filename: str
        The complete (relative or absolute) path to the file
    time_coord: str
        You can set it "seconds_elapsed" to have the time coordinate
        not as datetime value but as an float value counting the seconds
        from the beggining of the measurement.

    Output
    -------
    da: xarray DataArray
        all the recorded spectra with coordinates of each recording,
        along with the selected metadata as attributes.
    img: PIL image
        Returns `None` if no image was recorded
    """

    try:
        f = open(filename, "rb")
        if verbose:
            print(f'Reading the file: \"{filename.split("/")[-1]}\"\n')
    except IOError:
        raise IOError(f"File {filename} does not exist!")
    time_coord = kwargs.pop("time_coord", None)
    filesize = os.path.getsize(filename)
    params = dict()

    def _read(f=f, dtype=np.uint32, count=1):
        '''Reads bytes from binary file,
        with the most common values given as default.
        Returns the value itself if one value, or list if count > 1
        Note that you should do ".decode()"
        on strings to avoid getting strings like "b'string'"
        For further information, refer to numpy.fromfile() function
        '''
        if count == 1:
            return np.fromfile(f, dtype=dtype, count=count)[0]
        else:
            return np.fromfile(f, dtype=dtype, count=count)[0:count]

    def print_block_header(name, i, verbose=verbose):
        if verbose:
            print(f"\n{' Block : '+ name + ' ':=^80s}\n"
                  f"size: {blocks['BlockSizes'][i]},"
                  f"offset: {blocks['BlockOffsets'][i]}")

    blocks = dict()
    blocks["BlockNames"] = []
    blocks["BlockSizes"] = []
    blocks["BlockOffsets"] = []
    offset = 0
    # Reading all of the block names, offsets and sizes
    while offset < filesize - 1:
        header_dt = np.dtype([('block_name', '|S4'),
                              ('block_id', np.int32),
                              ('block_size', np.int64)])
        f.seek(offset)
        blocks["BlockOffsets"].append(offset)
        block_header = np.fromfile(f, dtype=header_dt, count=1)
        offset += block_header['block_size'][0]
        blocks["BlockNames"].append(block_header['block_name'][0].decode())
        blocks["BlockSizes"].append(block_header['block_size'][0])

    name = 'WDF1'
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i]+16)
#        TEST_WDF_FLAG = _read(f, np.uint64)
        params['WdfFlag'] = const.WDF_FLAGS[_read(f, np.uint64)]
        f.seek(60)
        params['PointsPerSpectrum'] = npoints = _read(f)
        # Number of spectra measured (nspectra):
        params['Capacity'] = nspectra = int(_read(f, np.uint64))
        # Number of spectra recorded (ncollected):
        params['Count'] = ncollected = int(_read(f, np.uint64))
        # Number of accumulations per spectrum:
        params['AccumulationCount'] = _read(f)
        # Number of elements in the y-list (>1 for image):
        params['YlistLength'] = _read(f)
        params['XlistLength'] = _read(f)  # number of elements in the x-list
        params['DataOriginCount'] = _read(f)  # number of data origin lists
        params['ApplicationName'] = _read(f, '|S24').decode()
        version = _read(f, np.uint16, count=4)
        params['ApplicationVersion'] = '.'.join(
            [str(x) for x in version[0:-1]]) +\
            ' build ' + str(version[-1])
        params['ScanType'] = const.SCAN_TYPES[_read(f)]
        params['MeasurementType'] = const.MEASUREMENT_TYPES[_read(f)]
        params['StartTime'] = convert_time(_read(f, np.uint64))
        params['EndTime'] = convert_time(_read(f, np.uint64))
        params['SpectralUnits'] = const.DATA_UNITS[_read(f)]
        laser_wavenumber = _read(f, '<f')
        params['LaserWaveLength'] = np.round(10e6 / laser_wavenumber, 2) \
            if laser_wavenumber else "Unspecified"
        # try:
        #     params['LaserWaveLength'] = np.round(10e6/_read(f, '<f'), 2)
        # except RuntimeWarrning as err:
        #     print(err)
        #     params['LaserWaveLength'] = "unknown"
        f.seek(240)
        params['Title'] = _read(f, '|S160').decode()

    if verbose:
        for key, val in params.items():
            print(f'{key:-<40s} : \t{val}')
        if nspectra != ncollected:
            print(f'\nATTENTION:\nNot all spectra were recorded\n'
                  f'Expected nspectra={nspectra},'
                  f'while ncollected={ncollected}'
                  f'\nThe {nspectra-ncollected} missing values'
                  f'will be replaced by zeros\n')

    def pad_if_unfinished(arr, count=params['Count'],
                          capacity=params['Capacity'], replace_value=np.nan):
        if count < capacity:
            arr[count:] = replace_value
        return arr

    name = 'WMAP'
    map_params = {}
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        m_flag = _read(f)
        map_params['MapAreaType'] = const.MAP_TYPES[m_flag]  # _read(f)]
        _read(f)
        map_params['InitialCoordinates'] = np.round(_read(f, '<f', count=3), 2)
        map_params['StepSizes'] = np.round(_read(f, '<f', count=3), 2)
        map_params['NbSteps'] = n_x, n_y, n_z = _read(f, np.uint32, count=3)
        map_params['LineFocusSize'] = _read(f)
    if verbose:
        for key, val in map_params.items():
            print(f'{key:-<40s} : \t{val}')

    name = 'DATA'
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        data_points_count = ncollected * npoints
        spectra = np.zeros((nspectra, npoints))  # container
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        spectra[:ncollected] = _read(f, '<f',
                                     count=data_points_count
                                     ).reshape(ncollected, npoints)
        if verbose:
            print(f'{"The number of spectra":-<40s} : \t{ncollected}')
            print(f'{"The number of points in each spectra":-<40s} : \t'
                  f'{npoints}')

    name = 'XLST'
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        params['XlistDataType'] = const.DATA_TYPES[_read(f)]
        params['XlistDataUnits'] = const.DATA_UNITS[_read(f)]
        x_values = _read(f, '<f', count=npoints)
    if verbose:
        print(f"{'The shape of the x_values is':-<40s} : \t{x_values.shape} ")
        print(f"*These are the \"{params['XlistDataType']}"
              f"\" recordings in \"{params['XlistDataUnits']}\" units")

    name = 'YLST'  # Not sure what's this about
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        params['YlistDataType'] = const.DATA_TYPES[_read(f)]
        params['YlistDataUnits'] = const.DATA_UNITS[_read(f)]
        y_values_count = int((blocks["BlockSizes"][i]-24)/4)
        if y_values_count > 1:
            y_values = _read(f, '<f', count=y_values_count)
            print(y_values)

    name = 'WHTL'  # This is where the image is
    img = None
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        img_bytes = _read(f, count=int((blocks["BlockSizes"][i]-16)/4))
        img = Image.open(io.BytesIO(img_bytes))

    # name = 'WXDB'  # Series of images
    # gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    # if len(gen) > 0:
    #     imgs = []
    #     img_sizes = []
    #     img_psets = dict()
    #     for i in gen:
    #         print_block_header(name, i)
    #         f.seek(blocks["BlockOffsets"][i] + 16)
    #         img_offsets = _read(f, dtype='u8', count=nspectra)
    #         for nn, j in enumerate(img_offsets):
    #             f.seek(int(j+blocks["BlockOffsets"][i]))
    #             size = _read(f)
    #             img_sizes.append(size)
    #             img_type = _read(f, dtype=np.uint8)
    #             img_flag = _read(f, dtype=np.uint8)
    #             img_key = _read(f, dtype=np.uint16)
    #             img_size = _read(f)
    #             img_length = _read(f)
    #             img_psets[nn] = {"img_type": img_type,
    #                              "img_flag": img_flag,
    #                              "img_key": img_key,
    #                              "img_size": img_size,
    #                              "img_length": img_length}

    name = 'ORGN'
    origin_labels = []
    origin_set_dtypes = []
    origin_set_units = []
    # origin_values = np.empty((params['DataOriginCount'], nspectra), dtype='<d')
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        coord_dict = dict()
        # coord_names = {3:"x", 4:"y", 5:"z", 6:"r"}
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        nb_origin_sets = _read(f)
        # The above is the same as params['DataOriginCount']
        for set_n in range(nb_origin_sets):
            data_type_flag = _read(f).astype(np.uint16)
            # not sure why I had to convert to uint16,
            # but if I just read it as uint32, I got rubbish sometimes
            origin_set_dtypes.append(const.DATA_TYPES[data_type_flag])
            coord_units = const.DATA_UNITS[_read(f)].lower()
            origin_set_units.append(coord_units)
            label = _read(f, '|S16').decode()
            origin_labels.append(label)

            if data_type_flag == 11:  # special case for reading timestamps
                recording_time = np.array(1e-7 *
                                          _read(f, np.uint64, count=nspectra)
                                          - 11644466400,
                                          dtype='datetime64[s]')
                # I had to add 2 hours to make it compatible with da.StartTime
                # othewise it was:  - 11644473600
                if time_coord == "seconds_elapsed":
                    recording_time = recording_time -\
                                     np.datetime64(params['StartTime'])
                    print(recording_time[2])
                    recording_time = np.round(
                                         recording_time.astype("float") * 1e-6,
                                         2)
                if recording_time.ndim == 0:  # for single scan measurement
                    recording_time = np.expand_dims(recording_time, 0)
                recording_time = pad_if_unfinished(
                                    recording_time,
                                    replace_value=recording_time[ncollected-1])

                coord_dict = {**coord_dict,
                              label: ("points", recording_time,
                                      {"units": coord_units}
                                      )
                              }
            else:
                coord_values = np.array(
                                  np.round(
                                      _read(f, '<d', count=nspectra),
                                      2))
            if data_type_flag not in [0, 11, 16, 17]:
                # 0:?
                # 11:Time - a special case already dealt with above
                # 16:Checksum - never saw anything useful recorded here
                # 17:Flags - same as 16, probably not used? - check with Renishaw?
                if coord_values.ndim == 0:  # if it's a single scan measurement
                    coord_values = np.expand_dims(coord_values, 0)
                coord_dict = {**coord_dict,
                              label: ("points", coord_values,
                                      {"units": coord_units}
                                      )
                              }
        if verbose:
            print(list(zip(origin_set_dtypes,
                           origin_set_units,
                           origin_labels)))

    if verbose:
        print('\n\n\n')
        print("coordinate", blocks)
    if params["Count"] != params["Capacity"]:
        warn(f"Not all spectra was recorded. \nExpected {nspectra}, "
             f"but only {ncollected} spectra were recorded.\n"
             f"The {nspectra-ncollected} missing spectra will be filled with "
             "zeros."
             "\n\nPlease bear in mind that working with such incomplete"
             " recordings might (and probably will) lead to odd results"
             " further down the pipeline.")

    da = xr.DataArray(spectra,
                      dims=("points", "RamanShifts"),
                      coords={**coord_dict,
                              "shifts": ("RamanShifts", x_values,
                                         {"units": "1/cm"}
                                         )
                              },
                      attrs={**params,
                             **map_params,
                             # **blocks,
                             "FileSize": hr_filesize(filesize)
                             }
                      )
    if len(map_params) > 0:
        if map_params["MapAreaType"] == "Slice":
            if ("R" in da.coords) and ("Z" in da.coords):
                scan_axes = np.array([2, 0])
                _coord_choice = ["R", "R", "Z"]
            else:
                scan_axes = np.array([1, 0])
                _coord_choice = ["X", "Y", "Z"]
        else:
            scan_axes = np.array([1, 0])
            _coord_choice = ["X", "Y", "Z"]
        scan_shape = tuple(da.attrs["NbSteps"][scan_axes])
        # scan_axes = np.argwhere(da.attrs["NbSteps"]>1).squeeze()
        # _coord_choice = ["R", "R", "Z"] if 2 in scan_axes else ["X", "Y", "Z"]
        # if 1 in scan_shape:
        #     da.attrs["MeasurementType"] = "Series like"
        col_coord = _coord_choice[scan_axes[1]]
        row_coord = _coord_choice[scan_axes[0]]
        da.attrs["ScanShape"] = scan_shape
        da.attrs["ColCoord"] = col_coord
        da.attrs["RowCoord"] = row_coord
    else:  # not a map scan
        da.attrs["ScanShape"] = (spectra.shape[0], 1)
        da.attrs["ColCoord"] = ""
        da.attrs["RowCoord"] = da.attrs["MeasurementType"]

    da = da.sortby(["shifts", "Time"])
    # Oddly enough, in case of slice scans
    # it appears that spectra aren't recorded with increasing timestamps (?!)
    if len(map_params) > 0:
        if map_params["MapAreaType"] != "Slice":
            # No matter the type of map scan, we want the same order
            if ("X" in da.coords.keys()) and\
               (0 in np.argwhere(da.attrs["NbSteps"] > 1)):
                da = da.sortby("X")
            if ("Y" in da.coords.keys()) and\
               (1 in np.argwhere(da.attrs["NbSteps"] > 1)):
                da = da.sortby("Y")
            else:
                pass
        else:
            try:
                if len(da.Z) > 1:
                    da = da.sortby("Z")
            except (KeyError, AttributeError):
                pass
    da.attrs["Folder name"], da.attrs["Filename"] = os.path.split(filename)

    return da, img
