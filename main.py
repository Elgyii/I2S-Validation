#!/usr/bin/env python3
# coding: utf-8
"""
Name:        Match-up main
Purpose:     Level-2 Data Match-up tool

authorship
__author__     = "Eligio Maure"
__license__    = ""
__version__    = "1.0.1"
__maintainer__ = "Eligio Maure"
__email__      = "maure at npec dot or dot jp"

Parts of the script are obtained from fd_matchup.py
by J.Scott on 2016/12/12 (joel.scott@nasa.gov)

Comments/questions:
  email: maure at npec dot or dot jp (E. R. Maure)
2020/10/07
"""
import logging
import re
import sys
import time
from datetime import (datetime, timedelta)
from pathlib import Path

import numpy as np
from dateutil.parser import parse
from pandas import DataFrame

from sutils import (MatchUpError, FileSanity)
from smatch import MatchUp
from sget import (getfile, search, UrlParser, SATELLITES)


def fmt_time(hms: str, debug, logger):
    """Format time str to h m s
       t = re.search(r"(\b\d{1,2}\b):(\b\d{1,2}\b)", hms)
    """
    if debug:
        logger.info(hms)
    # Repetition qualifiers: (*, +, ?, {m,n}, etc)
    t = re.search(r"([0-9]{1,2}|[1-2][0-3]{1,2}):"
                  r"([0-9]{1,2}|[1-5][0-9]{1,2}):"
                  r"([0-9]{1,2}|[1-5][0-9]{1,2})", hms)
    if t:
        return int(t.group(1)), int(t.group(2)), int(t.group(3))
    # \b is defined as the boundary between a \w and \W
    #    represents the backspace character
    t = re.search(r"([0-9]{1,2}|[1-2][0-3]{1,2}):"
                  r"([0-9]{1,2}|[1-5][0-9]{1,2})", hms)
    return int(t.group(1)), int(t.group(2)), 0


def fmt_date(ymd: str, debug, logger):
    """ function to format datetime from re compiled result; returns datetime
    There is ambiguity between month/day when order is not known.
    Searching formats default YYYYMMDD | YYYY-MM-DD | YYYY/MM/DD
    """

    # ----------------
    # year/month/day
    # default expected
    # ----------------
    if debug:
        logger.info(ymd)
    fmt = re.search(r"(\d{4})"
                    r"(1[0-2]|0[1-9])"
                    r"(3[01]|[12][0-9]|0[1-9])", ymd)
    if fmt:
        return int(fmt.group(1)), int(fmt.group(2)), int(fmt.group(3))
    fmt = re.search(r"(\d{4})/"
                    r"(1[0-2]|0[1-9])/"
                    r"(3[01]|[12][0-9]|0[1-9])", ymd)
    if fmt:
        return int(fmt.group(1)), int(fmt.group(2)), int(fmt.group(3))
    fmt = re.search(r"(\d{4})-"
                    r"(1[0-2]|0[1-9])-"
                    r"(3[01]|[12][0-9]|0[1-9])", ymd)
    if fmt:
        return int(fmt.group(1)), int(fmt.group(2)), int(fmt.group(3))

    # --------------
    # month/day/year
    # --------------
    fmt = re.search(r"(1[0-2]|0[1-9])"
                    r"(3[01]|[12][0-9]|0[1-9])"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(1)), int(fmt.group(2))
    fmt = re.search(r"(1[0-2]|0[1-9])/"
                    r"(3[01]|[12][0-9]|0[1-9])/"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(1)), int(fmt.group(2))
    fmt = re.search(r"(1[0-2]|0[1-9])-"
                    r"(3[01]|[12][0-9]|0[1-9])-"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(1)), int(fmt.group(2))

    # --------------
    # month/day/year
    # --------------
    fmt = re.search(r"(1[0-2]|[1-9])"
                    r"(3[01]|[12][0-9]|[1-9])"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(1)), int(fmt.group(2))
    fmt = re.search(r"(1[0-2]|[1-9])/"
                    r"(3[01]|[12][0-9]|[1-9])/"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(1)), int(fmt.group(2))
    fmt = re.search(r"(1[0-2]|[1-9])-"
                    r"(3[01]|[12][0-9]|[1-9])-"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(1)), int(fmt.group(2))

    # --------------
    # day/month/year
    # --------------
    fmt = re.search(r"(3[01]|[12][0-9]|0[1-9])"
                    r"(1[0-2]|0[1-9])"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(2)), int(fmt.group(1))
    fmt = re.search(r"(3[01]|[12][0-9]|0[1-9])/"
                    r"(1[0-2]|0[1-9])/"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(2)), int(fmt.group(1))
    fmt = re.search(r"(3[01]|[12][0-9]|0[1-9])-"
                    r"(1[0-2]|0[1-9])-"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(2)), int(fmt.group(1))

    # --------------
    # day/month/year
    # --------------
    fmt = re.search(r"(3[01]|[12][0-9]|[1-9])"
                    r"(1[0-2]|[1-9])"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(2)), int(fmt.group(1))
    fmt = re.search(r"(3[01]|[12][0-9]|[1-9])/"
                    r"(1[0-2]|[1-9])/"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(2)), int(fmt.group(1))
    fmt = re.search(r"(3[01]|[12][0-9]|[1-9])-"
                    r"(1[0-2]|[1-9])-"
                    r"(\d{4})", ymd)
    if fmt:
        return int(fmt.group(3)), int(fmt.group(2)), int(fmt.group(1))

    fmt = parse(ymd)
    return fmt.year, fmt.month, fmt.day


def fmt_datetime(ymd: str, hms: str, debug: bool, logger):
    """ date and time conversion to datetime objects """

    hr, mnt, sec = fmt_time(hms=hms, debug=debug, logger=logger)
    yr, mon, day = fmt_date(ymd=ymd, debug=debug, logger=logger)

    return datetime(year=yr,
                    month=mon,
                    day=day,
                    hour=hr,
                    minute=mnt,
                    second=sec)


def check_ifile(filename: Path, debug: bool, logger):
    """
     function to verify text file exists, is valid, and has correct fields; returns data structure

     Parameters
     ----------
     filename: Path
        input filename to read
     debug: bool
     logger: logging

     Returns
     -------
     data_frame: DataFrame
        DataFrame with user input information for running the match-ups
    """
    from pandas import read_csv

    if filename.is_file():
        try:
            ds = read_csv(filename
                          , sep=','
                          , encoding='utf-8'
                          , skip_blank_lines=True
                          , low_memory=False
                          , parse_dates=False
                          )
        except UnicodeDecodeError:
            ds = read_csv(filename
                          , sep=','
                          , encoding='shift-jis'
                          , skip_blank_lines=True
                          , low_memory=False
                          , parse_dates=False
                          )
    else:
        info = f'ERROR: invalid --text_file specified. Does: {filename.name} exist?'
        if debug:
            logger.info(info)
        raise MatchUpError(info)

    fields = [
        'date'
        , 'year'
        , 'month'
        , 'day'
        , 'time'
        , 'hour'
        , 'minute'
        , 'second'
        , 'lon'
        , 'lat'
        , 'datetime']

    columns = {}
    update = columns.update
    [update({col: field.title()})
     for field in fields
     for col in ds.columns
     if field in col.lower()]

    if debug:
        logger.info(f'{columns, ds.shape}\n{ds}')

    if len(columns):
        ds.rename(columns=columns, inplace=True)

    date_time = None
    idx = [i for i, col in enumerate(ds.columns)
           if col.lower() in ['date', 'year', 'month', 'day']]
    idx += [i for i, col in enumerate(ds.columns)
            if col.lower() in ['time', 'hour', 'minute', 'second']]

    if debug:
        logger.info(ds.columns)
    info = 'missing fields in text file. File must contain date/time, date/hour/' \
           'minute/second, year/month/day/time, OR year/month/day/hour/minute/second'
    if (len(idx) == 0) and not ('Datetime' in ds.columns):
        if debug:
            logger.info(info)
        raise MatchUpError(info)

    if 'Datetime' in ds.columns:
        return ds

    # columns = ds.columns[idx]
    if len(idx) == 2:
        cols = ['Date', 'Time']
        date_time = [fmt_datetime(ymd=ymd,
                                  hms=hms,
                                  logger=logger,
                                  debug=debug)
                     for ymd, hms in ds.loc[:, cols].values]

    if len(idx) == 5:
        cols = ['Year', 'Month', 'Day', 'Hour', 'Minute']
        date_time = [datetime(
            year=int(yr), month=int(mon), day=int(day),
            hour=int(hr), minute=int(mnt))
            for yr, mon, day, hr, mnt in ds.loc[:, cols].values]

    if len(idx) == 6:
        cols = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']
        date_time = [datetime(
            year=int(yr), month=int(mon), day=int(day),
            hour=int(hr), minute=int(mnt), second=int(sec))
            for yr, mon, day, hr, mnt, sec in ds.loc[:, cols].values]

    ds['Datetime'] = date_time
    # logger.info(ds.columns)
    # logger.info(renamed)
    # if len(renamed) > 0:
    #     ds.rename(columns=renamed, inplace=True)
    return ds


def check_geo(ds: DataFrame, logger: logging, debug: bool):
    """ function to verify lat/lon exist in SB file's data structure """
    lons, lats = [], []
    try:
        for lat, lon in zip(ds['Lat'], ds['Lon']):
            if debug:
                logger.info(f'Lat: {lat} | Lon: {lon}')
            validate_lat(lat=float(lat))
            validate_lon(lon=float(lon))
            lats.append(float(lat))
            lons.append(float(lon))
    except Exception as e:
        raise MatchUpError('missing fields in INPUT file. File must contain lat,lon')
    ds['Lon'] = lons
    ds['Lat'] = lats
    return ds


def validate_lat(lat: float):
    """ function to verify lat range """
    if abs(lat) > 90.0:
        raise MatchUpError(f'invalid latitude: all LAT values MUST '
                           f'be between -90/90N deg. Received: {lat}')
    return lat


def validate_lon(lon: float):
    """ function to verify lon range """
    if abs(lon) > 180.0:
        raise MatchUpError(f'invalid longitude: all LON values MUST be between -180/180E deg. '
                           f'Received: {lon}')
    return lon


def skip(mission: str, day: int):
    """
    Checks if sensor is within its lifespan
    :param mission:
    :param day:
    :return:
    """
    return (SATELLITES[mission]['PERIOD_START'] > day) or \
           (day > SATELLITES[mission]['PERIOD_END'])


def smat_main(ifile: Path
              , ofile: Path
              , odir: Path
              , parse_vars: dict
              , debug: bool
              , logger):
    """handles inputs from a file"""
    from math import isnan

    start = time.perf_counter()

    sat = parse_vars['sat'][0]
    sst_flag = parse_vars['sst_flag'][0].upper() \
        if parse_vars['data_type'][0] == 'sst' else None

    case = 'csw' if sat == 'sgli' else 'cmr'
    dx = dy = .01

    max_time_diff = parse_vars.pop('max_time_diff')[0]
    twin_hmn = -1 * int(max_time_diff)
    twin_mmn = -60 * (max_time_diff - int(max_time_diff))
    twin_hmx = 1 * int(max_time_diff)
    twin_mmx = 60 * (max_time_diff - int(max_time_diff))

    data_frame = check_ifile(filename=ifile, debug=debug, logger=logger)
    if debug:
        logger.info(data_frame.columns)
    url_parser = UrlParser(tim_min=twin_hmn,
                           tim_max=twin_hmx,
                           **{key: val[0]
                              for key, val in parse_vars.items()
                              if val is not None})

    file_sanity = FileSanity(check_list=[]
                             , instrument=''
                             , logger=logger
                             , host=parse_vars['host'][0])

    if debug:
        logger.info(data_frame)
    dtype = parse_vars['data_type'][0]
    data_frame = check_geo(ds=data_frame, logger=logger, debug=debug)
    data_frame.replace(np.nan, '-999', inplace=True)

    prc = f'{0:.2f}'

    cntl_file = odir.joinpath('control_list.txt')
    with open(cntl_file, 'a') as txt:
        txt.writelines('')

    total = data_frame.shape[0]
    dec, iter_counter = len(f'{total}'), 0
    dates = np.asarray([d.strftime('%F') for d in data_frame.loc[:, 'Datetime']])
    unique_days = np.unique(dates)
    match_up_file = f'No valid satellite match-ups found for any lat/lon/time pairs in {ifile}'

    # ============ PARAMS ==============
    host = parse_vars.pop('host')[0]
    kwargs = {
        'host': host
        , 'user': parse_vars.pop('user')[0]
        , 'email': parse_vars.pop('email')[0]
        , 'satellite': f'{SATELLITES[sat]["INSTRUMENT"]}/'
                       f'{SATELLITES[sat]["PLATFORM"]}'
        , 'max_time_diff': max_time_diff
        , 'csv_file': ofile
        , 'data_type': parse_vars.pop('data_type')[0]
        , 'variables': parse_vars.pop('variables')
        , 'pixel_window_size': parse_vars.pop('pixel_window_size')[0]
        , 'min_valid_pixels': parse_vars.pop('min_valid_pixels')[0]
        , 'l2_bits': parse_vars.pop('l2_bits')[0]
        , 'sst_flag': sst_flag
        , 'sst_quality_level': parse_vars.pop('sst_quality_level')[0]
        , 'datum': 'WGS84'
        , 'max_iter': 200
        , 'logger': logger
    }

    mode, tds, header_saved = 'w', unique_days.size, False
    tec, found = len(f'{tds}'), 0
    # Process files on daily basis to avoid too much data download
    for d, day in enumerate(unique_days):
        info = f'Day: {day}, {(d + 1):{tec}} in {tds}'
        logger.info(f'{"*" * len(info)}\n{info}\n{"*" * len(info)}')

        match = data_frame.loc[dates == day, :].copy()
        unmatch = data_frame.loc[dates == day, :].copy()

        match.reset_index(drop=True, inplace=True)
        unmatch.reset_index(drop=True, inplace=True)

        match['sat_files'] = [[]] * match.shape[0]
        unmatch['save_empty'] = [True] * unmatch.shape[0]
        file: Path = Path('.')

        for row, series in match.iterrows():
            # logger.debug(f'Row: {row} | {series}')
            iter_counter += 1
            prc = f'{(iter_counter / total * 100):.2f}'
            lon, lat, dt = series.Lon, series.Lat, series.Datetime

            url_parser.tim_min = dt + timedelta(hours=twin_hmn,
                                                minutes=twin_mmn)
            url_parser.tim_max = dt + timedelta(hours=twin_hmx,
                                                minutes=twin_mmx)

            count = f'FileSearch: {iter_counter:0{dec}} ({prc}%) OUT-OF {total}'
            st_msg = f'     Start: {url_parser.tim_min}'
            n = max(len(st_msg), len(count))

            message = f'       Lon: {lon}\n' \
                      f'       Lat: {lat}\n' \
                      f'{st_msg}\n' \
                      f'       End: {url_parser.tim_max}\n' \
                      f'{count}'

            if skip(mission=sat, day=dt.toordinal()):
                logger.info(f'{message}\nOutSatTimeRange\n{"=" * n}\n')
                continue

            if isnan(lat) or isnan(lon):
                logger.info(f'{message}\nNoValid: LonLat\n{"=" * n}\n')
                continue

            validate_lon(lon=lon)
            validate_lat(lat=lat)

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
            if debug:
                logger.info(url)
                if host == 'npec':
                    print(url)

            try:
                content = search(url=url, sen=sat, debug=debug, sst_flag=sst_flag)
            except ConnectionResetError:
                logger.info(time.ctime())
                raise
                # ------------------
            # Download the files
            # ------------------
            if content:
                files = getfile(content=content
                                , out_dir=odir
                                , logger=logger
                                , case=case)
            else:
                logger.warning('WARNING: No matching granules found for the row.\n'
                               'Continuing to search for granules from the rest of the input file...\n')
                if host == 'npec':
                    print('WARNING: No matching granules found for the row.\n'
                          'Continuing to search for granules from the rest of the input file...',
                          file=sys.stderr)
                continue

            with open(cntl_file, 'r') as txt:
                control_list = list(filter(None, txt.readlines()))

            file_sanity.check_list = list(set(files))
            file_sanity.instrument = sat
            file_sanity.control_list = control_list
            files = file_sanity.check()

            if len(files):
                with open(cntl_file, 'w') as txt:
                    previous = ''.join(control_list)
                    current = ''.join([f'{Path(f).name}:OK\n' for f in files])
                    txt.writelines(f'{previous}\n{current}\n')

                if debug:
                    logger.debug(f'Row: {row}\nIDX\n{match}\nDF\n{match}')

            if len(files) > 0:
                match.at[row, 'sat_files'] = files
                unmatch.at[row, 'save_empty'] = False
                file = Path(files[0])
        # ---------------
        # Get the matchup
        # ---------------
        mode = 'w' if header_saved is False else 'a'
        if file.is_file():
            args = match, file, mode, prc,
        else:
            args = match, mode, prc
        cfm = MatchUp(
            *args, **kwargs.copy()
        ).get()
        found += cfm
        header_saved = True

        # ---------------------
        # Del current day files
        # ---------------------
        for i, series in match.iterrows():
            for f in series.sat_files:
                f.unlink(missing_ok=True)

    logger.info(f'{found} match-ups saved to: "{ofile}"')
    if host == 'npec':
        print(f'{found} match-ups saved to "{ofile}"')
    # -----------------
    # Return the result
    # -----------------
    time_elapsed = (time.perf_counter() - start)
    hrs = int(time_elapsed // 3600)
    mnt = int(time_elapsed % 3600 // 60)
    sec = int(time_elapsed % 3600 % 60)
    logger.info(f'Processing Time:{hrs:3} hrs {mnt:3} min{sec:3} sec')
    if found == 0:
        return match_up_file
    return ofile
