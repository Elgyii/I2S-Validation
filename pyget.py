#!/usr/bin/env python3
# coding: utf-8
"""
Name:        main
Purpose:     Level-2 Data Match-up tool

authorship
__author__     = "Eligio Maure"
__license__    = ""
__version__    = "1.0.1"
__maintainer__ = "Eligio Maure"
__email__      = "maure at npec dot or dot jp"

Comments/questions:
  email: maure at npec dot or dot jp (E. R. Maure)
2020/10/07
"""

import logging
import os
import subprocess
import sys
import textwrap
import time
from datetime import timedelta
from math import isnan
from pathlib import Path

import coloredlogs
import numpy as np

import sget
import sutils

__version__ = '1.0.1'

USER = os.environ['USER'] if 'USER' in os.environ.keys() else 'None'
LOGGER = logging.getLogger(__name__)
LEVEL = logging.DEBUG
LOGGER.setLevel(level=LEVEL)
DEBUG = False


def remove_temp_dir(dirname: Path):
    # --------------------------------
    # Remove temporary download folder
    # --------------------------------
    abspath = dirname.absolute()
    cmd = f'rmdir /Q /S {abspath} >nul' \
        if (os.name == 'nt') else \
        f"rm -r -f {abspath} 2> /dev/null"
    subprocess.call(cmd, shell=True)
    return 0


def get_logger(name: str = None):
    logger_dir = Path().cwd().joinpath('logs')
    logger_fmt = '\n%(module)s.%(funcName)s:' \
                 '%(lineno)d\n%(message)s'
    encoded_styles = 'DEBUG=magenta;' \
                     'info=green;' \
                     'warning=yellow;' \
                     'critical=red,bold;' \
                     'exception=blue'

    if not logger_dir.is_dir():
        logger_dir.mkdir()

    full_name = name if name is None else \
        logger_dir.joinpath(f'{name}.log')
    # console = False if USER == 'npec' else True

    # ch = None
    # console handler for print logs
    ch = logging.StreamHandler()
    ch.setLevel(LEVEL)

    # formatter for the handlers
    formatter = logging.Formatter(fmt=logger_fmt)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    LOGGER.addHandler(ch)

    if name is None:
        LOGGER.handlers = [ch]
    else:
        fh = logging.FileHandler(
            full_name
            , encoding='utf-8'
            , mode='a')

        fh.setLevel(LEVEL)
        formatter = logging.Formatter(
            fmt='\n%(module)s.%(funcName)s:%(lineno)d | '
                ' %(asctime)s\n%(message)s')

        fh.setFormatter(formatter)
        LOGGER.handlers = [fh, ch]

    coloredlogs.install(
        level=LEVEL
        , level_styles=coloredlogs.parse_encoded_styles(encoded_styles)
        , fmt=logger_fmt
        , logger=LOGGER
    )
    return LOGGER


def file_get(output_dir: Path, text_file: str, params: dict, logger: logging):
    start = time.perf_counter()

    sat = params['sat'][0]
    sst_flag = params['sst_flag'][0]
    case = 'csw' if sat == 'sgli' else 'cmr'
    dx = dy = .01

    max_time_diff = params.pop('max_time_diff')[0]
    twin_hmn = -1 * int(max_time_diff)
    twin_mmn = -60 * (max_time_diff - int(max_time_diff))
    twin_hmx = 1 * int(max_time_diff)
    twin_mmx = 60 * (max_time_diff - int(max_time_diff))

    data_frame = sutils.check_ifile(filename=Path(text_file)
                                    , debug=DEBUG
                                    , logger=logger)
    if DEBUG:
        logger.info(data_frame.columns)
    url_parser = sutils.UrlParser(tim_min=twin_hmn,
                                  tim_max=twin_hmx,
                                  **{key: val[0]
                                     for key, val in params.items()
                                     if val is not None})

    file_sanity = sutils.FileSanity(check_list=[]
                                    , instrument=''
                                    , logger=logger
                                    , host=USER)

    if DEBUG:
        logger.info(data_frame)
    dtype = params['data_type'][0]
    data_frame = sutils.check_geo(ds=data_frame, logger=logger, debug=DEBUG)
    data_frame.replace(np.nan, '-999', inplace=True)

    prc = f'{0:.2f}'

    total = data_frame.shape[0]
    dec, iter_counter = len(f'{total}'), 0
    dates = np.asarray([d.strftime('%F') for d in data_frame.loc[:, 'Datetime']])
    unique_days = np.unique(dates)

    mode, tds, header_saved = 'w', unique_days.size, False
    tec, found = len(f'{tds}'), 0
    # Process files on daily basis to avoid too much data download

    for row, series in data_frame.iterrows():
        # logger.debug(f'Row: {row} | {series}')
        iter_counter += 1
        prc = f'{(iter_counter / total * 100):.2f}'
        lon, lat, dt = series.Lon, series.Lat, series.Datetime

        url_parser.tim_min = dt + timedelta(hours=twin_hmn, minutes=twin_mmn)
        url_parser.tim_max = dt + timedelta(hours=twin_hmx, minutes=twin_mmx)

        count = f'FileSearch: {iter_counter:0{dec}} ({prc}%) OUT-OF {total}'
        st_msg = f'     Start: {url_parser.tim_min}'
        n = max(len(st_msg), len(count))

        message = f'       Lon: {lon}\n' \
                  f'       Lat: {lat}\n' \
                  f'{st_msg}\n' \
                  f'       End: {url_parser.tim_max}\n' \
                  f'{count}'

        if sutils.skip(mission=sat, day=dt.toordinal()):
            logger.info(f'{message}\nOutSatTimeRange\n{"=" * n}\n')
            continue

        if isnan(lat) or isnan(lon):
            logger.info(f'{message}\nNoValid: LonLat\n{"=" * n}\n')
            continue

        sutils.validate_lon(lon=lon)
        sutils.validate_lat(lat=lat)

        # -----------------------------------
        logger.info(f'{message}\n{"=" * n}')
        # -----------------------------------

        if (sat != 'sgli') and (dtype == 'sst'):
            url_parser.tim_min = dt

        if (sat == 'sgli') or (dtype == 'sst'):
            url_parser.slat = lat - dy
            url_parser.elat = lat + dy
            url_parser.slon = lon - dx
            url_parser.elon = lon + dx
        else:
            url_parser.slon = lon
            url_parser.slat = lat

        url = url_parser.csw_url() if sat == 'sgli' else url_parser.cmr_point()
        if DEBUG:
            logger.info(url)

        try:
            content = sget.search(url=url
                                  , debug=DEBUG
                                  , sst_flag=sst_flag
                                  , sen=sat)
        except ConnectionResetError:
            logger.info(time.ctime())
            raise
            # ------------------
        # Download the files
        # ------------------
        if content:
            files = sget.getfile(content=content
                                 , out_dir=output_dir
                                 , logger=logger
                                 , case=case)
        else:
            logger.warning('WARNING: No matching granules found for the row.\n'
                           'Continuing to search for granules from the rest of the input file...\n')
            continue

        file_sanity.check_list = list(set(files))
        file_sanity.instrument = sat
        file_sanity.check()

    # -----------------
    # Return the result
    # -----------------
    time_elapsed = (time.perf_counter() - start)
    hrs = int(time_elapsed // 3600)
    mnt = int(time_elapsed % 3600 // 60)
    sec = int(time_elapsed % 3600 % 60)
    logger.info(f'Processing Time:{hrs:3} hrs {mnt:3} min{sec:3} sec')
    return


def cli_main():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''\
      This program uses Earthdata Search Tool (https://cmr.earthdata.nasa.gov/search/) based on Common
      Metadata Repository (CMR) and GPortal's Search Tool (https://gportal.jaxa.jp/csw/csw) based on
      comparative web search (CSW) system to find collections from the data provided by the NASA OB.DAAC
      and JAXA GPortal. The tool finds satellite granule names given an satellite/instrument and
      lat/lon/time point or range information.

      Outputs:
         1a) a list of OB.DAAC L2 satellite file granule names that contain the input criteria, per the CMR's records.
         1b) a list of GPortal L2 satellite file granule names that contain the input criteria, per the CSW's records.
         2) a list of public download links to fetch the matching satellite file granules, per the CMR's/CSW's records.

      Inputs:
        The argument-list is a set of --keyword value pairs (see optional arguments below).

        * Compatibility: This script was developed with Python 3.7.

         License:
           /*=====================================================================*/
                            NASA Goddard Space Flight Center (GSFC) 
                    Software distribution policy for Public Domain Software

            The fd_matchup.py code is in the public domain, available without fee for 
            educational, research, non-commercial and commercial purposes. Users may 
            distribute this code to third parties provided that this statement appears
            on all copies and that no charge is made for such copies.

            NASA GSFC MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THE SOFTWARE
            FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED
            WARRANTY. NEITHER NASA GSFC NOR THE U.S. GOVERNMENT SHALL BE LIABLE FOR
            ANY DAMAGE SUFFERED BY THE USER OF THIS SOFTWARE.
           /*=====================================================================*/
           
      ''', epilog=textwrap.dedent('''\
        Type python pyget.py --help for details.
        --------
        Examples usage calls:
            python pyget.py --sat=modisa --data_type=IOP --text_file=get_modisa.csv --max_time_diff=1
            python pyget.py --sat=sgli --data_type=Rrs --text_file=get_sgli.csv 
            python pyget.py --sat=sgli --data_type=OC --text_file=get_sgli.csv 
            python pyget.py --sat=modisa --data_type=OC --text_file=get_modisa.csv 
                                           '''), add_help=True)

    parser.add_argument('--sat', choices=['sgli', 'modisa', 'viirsn', 'viirsj',
                                          'goci', 'czcs', 'octs', 'seawifs'],
                        type=str, nargs=1, required=True, help='''\
      String specifier for satellite platform/instrument

      Valid options are:
      -----------------
      sgli    = SGLI on GCOM-C
      viirsn  = VIIRS on NPP
      viirsj  = VIIRS on JPSS1
      modisa  = MODIS on AQUA
      goci    = GOCI on COMS
      czcs    = CZCS on Nimbus-7
      seawifs = SeaWiFS on OrbView-2
      octs    = OCTS on ADEOS-I
      ''')

    parser.add_argument('--data_type', nargs=1, type=str, default=(['*']),
                        choices=['OC', 'IOP', 'Rrs', 'SST'], help='''\
      OPTIONAL: String specifier for satellite data type
      Default behavior returns all product suites

      Valid options are:
      -----------------
      oc  = Returns OC (ocean color) product suite. Returns (CDOM, CHLA, TSM) if --sat == sgli
      iop = Returns IOP (inherent optical properties) product suite. Raises error if --sat == sgli
      rrs = Returns RRS (remote sensing reflectance) product suite from sgli
      sst = Returns SST product suite (including SST4 where applicable)
      ''')

    parser.add_argument('--text_file', nargs=1, type=str, required=True, help='''\
      Valid Text (csv) file name
      File must contain, at least, latitude, longitude, date and time information as fields.
      ''')

    parser.add_argument('--max_time_diff', nargs=1, default=([3.0]), type=float, help=('''\
      Maximum time difference between satellite and in situ point
      OPTIONAL: default value +/-3 hours
      Valid range: decimal number of hours (0-36)
      '''))

    parser.add_argument('--output_dir', nargs=1, type=str, default=([os.getcwd()]), help='''\
      OPTIONAL: output directory for the matchup file 
      Use this flag to save the output data to a separate directory from current working dir
      ''')

    parser.add_argument('--sst_flag', nargs=1, default=(['n']), choices=['4', 'd', 'n'], type=str, help=('''\
      SST flag, whether to use day or nighttime SST
      OPTIONAL: default value n (nighttime)
      Valid values: 4: short-wave (3-4 µm) thermal radiation SST
                    d: daytime SST
                    n: nighttime SST
      Use with --data_type=SST
      '''))

    parse_args = parser.parse_args()
    parse_vars = vars(parse_args)
    parse_vars['data_type'] = [parse_vars['data_type'][0].lower()]

    # logger_name = '_'.join([parse_vars['user'][0], datetime.today().strftime('%Y%jT%H%M%S')])
    logger = get_logger()

    if parse_vars['max_time_diff'][0] < 0 or parse_vars['max_time_diff'][0] > 36:
        info = f'invalid --max_time_diff value provided. Please specify a value between ' \
               f'0 and 36 hours. Received --max_time_diff = {parse_vars["max_time_diff"][0]}'
        (logger.exception(info) if DEBUG else parser.error(info)) \
            if USER == 'None' else (print(info, file=sys.stderr) if DEBUG else parser.error(info))

    if parse_vars["sst_flag"][0] not in ('4', 'd', 'n'):
        info = "invalid --sst_flag specified, please type 'python smatpy.py -h' for details"
        (logger.exception(info) if DEBUG else parser.error(info)) \
            if USER == 'None' else (print(info, file=sys.stderr) if DEBUG else parser.error(info))

    output_dir = Path(parse_vars['output_dir'][0])
    if not output_dir.is_dir():
        output_dir.mkdir()
    text_file = parse_vars.pop('text_file')[0]

    file_get(output_dir=output_dir
             , params=parse_vars
             , text_file=text_file
             , logger=logger)
    return


if __name__ == "__main__":
    cli_main()
