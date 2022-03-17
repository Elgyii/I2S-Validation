import os
import subprocess

import h5py
import numpy as np
from netCDF4 import Dataset
from numpy.ma import masked_array


def file_check(file: str, instrument: str, sds=None) -> masked_array:
    """Checks for swath data file integrity

    Parameters
    ----------
    file: str
        path of the swath file to be checked
    instrument: str
        name of the sensor being checked
    sds: [optional]: None
        initialization of the return value

    Returns
    -------
        np.ma.array
            data from input swath file or None if the file is corrupt/cannot be read
    """

    basename = os.path.basename(file)
    if instrument in ('octs', 'seawifs', 'modisa', 'viirsn', 'viirsj', 'goci'):
        key = 'chlor_a'
        if 'SST' in basename:
            key = 'sst'
        with Dataset(file) as dst:
            sds = dst.groups['geophysical_data'][key][:]
        return sds

    if instrument == 'sgli':
        key = 'CHLA'
        if 'NWLR' in basename:
            key = 'NWLR_412'
        if 'SST' in basename:
            key = 'SST'

        with h5py.File(file, 'r') as dst:
            sds = dst[f'/Image_data/{key}'][:]
            mask = np.equal(sds, dst[f'/Image_data/{key}'].attrs['Error_DN'][0])
            sds = np.ma.masked_array(sds, dtype=np.float32, fill_value=np.float32(-32767))
            sds.mask = mask
        return sds
    return sds


def check(check_list: list, instrument: str, logger=None) -> list:
    """Checks for swath data file integrity
    Sometimes there are empty files that need to be taken care of before mapping

    Parameters
    ----------
    check_list: list
        list of swath files to be checked
    instrument: str
        name of the sensor being checked
    logger: [Optional]: logging
        corrupt files is displayed using logger if supplied.

    Returns
    -------
        list
            list of swath files that pass this checking
    """

    keep_files = []
    append = keep_files.append
    cmd = 'del /f {file}' if os.name == 'nt' else 'rm -f {file}'

    for i, check_this in enumerate(check_list):
        check_file = os.path.abspath(check_this)
        bsn = os.path.basename(check_file)
        bsd = os.path.dirname(check_file)

        if not (bsn.endswith('.nc')
                or bsn.endswith('.h5')):
            continue

        try:
            data = file_check(file=check_file, instrument=instrument)
        except Exception as exc:
            if os.path.isfile(check_file):
                subprocess.call(cmd.format(file=bsn), cwd=bsd, shell=True)
            if logger:
                logger.exception(f'\tFile#: {(i + 1): 3d} | {bsn} | {instrument}\n{exc}')
            continue

        if data is None:
            subprocess.call(cmd.format(file=bsn), cwd=bsd, shell=True)
            if logger:
                logger.warning(f'\tFile#: {(i + 1): 3d} | {bsn}: BadFile, removed')
            continue

        if data[~data.mask].size == 0:
            subprocess.call(cmd.format(file=bsn), cwd=bsd, shell=True)
            if logger:
                logger.warning(f'\tFile#: {(i + 1): 3d} | {bsn}: Empty, removed')

        if data[~data.mask].size > 0:
            if logger:
                logger.info(f'\tFile#: {(i + 1): 3d} | {bsn}: Pass')
            append(check_file)
    return keep_files
