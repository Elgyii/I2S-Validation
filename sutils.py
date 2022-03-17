#!/usr/bin/env python3
# coding: utf-8
"""
Name:        File Utils
Purpose:     Level-2 Data Match-up tool

authorship
__author__     = "Eligio Maure"
__license__    = ""
__version__    = "1.0.1"
__maintainer__ = "Eligio Maure"
__email__      = "maure at npec dot or dot jp"
__status__     = "Routine data processing on AWS"

Parts of the script are obtained from fd_matchup.py
by J.Scott on 2016/12/12 (joel.scott@nasa.gov)

Comments/questions:
  email: maure at npec dot or dot jp (E. R. Maure)
2020/10/07
"""
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from dateutil.parser import parse
from netCDF4 import (Dataset)
from netCDF4 import date2num
from numpy.ma import masked_array
from pandas import DataFrame
from pyhdf.SD import (SD, SDC)

# dictionary of lists of CMR platform, instrument, collection names
SATELLITES = {
    'czcs': {'INSTRUMENT': 'CZCS',
             'PLATFORM': 'Nimbus-7',
             'SEARCH': 'CZCS_L2_',
             'SENSOR': '',
             'PERIOD_END': datetime(1978, 10, 30).toordinal(),
             'PERIOD_START': datetime(1986, 6, 22).toordinal()},

    'goci': {'INSTRUMENT': 'GOCI',
             'PLATFORM': 'COMS',
             'SEARCH': 'GOCI_L2_',
             'SENSOR': '',
             'PERIOD_END': datetime(2021, 4, 1).toordinal(),
             'PERIOD_START': datetime(2011, 4, 1).toordinal()},

    'meris': {'INSTRUMENT': 'MERIS',
              'PLATFORM': 'ENVISAT',
              'SEARCH': 'MERIS_L2_',
              'SENSOR': 'merr',
              'PERIOD_END': datetime(2012, 4, 8).toordinal(),
              'PERIOD_START': datetime(2002, 4, 29).toordinal()},

    'modisa': {'INSTRUMENT': 'MODIS',
               'PLATFORM': 'AQUA',
               'SEARCH': 'MODISA_L2_',
               'SENSOR': 'amod',
               'PERIOD_END': datetime.today().toordinal(),
               'PERIOD_START': datetime(2002, 7, 4).toordinal()},

    'modist': {'INSTRUMENT': 'MODIS',
               'PLATFORM': 'TERRA',
               'SEARCH': 'MODIST_L2_',
               'SENSOR': 'tmod',
               'PERIOD_END': datetime.today().toordinal(),
               'PERIOD_START': datetime(2000, 2, 24).toordinal()},

    'octs': {'INSTRUMENT': 'OCTS',
             'PLATFORM': 'ADEOS-I',
             'SEARCH': 'OCTS_L2_',
             'SENSOR': '',
             'PERIOD_END': datetime(1997, 6, 29).toordinal(),
             'PERIOD_START': datetime(1996, 10, 31).toordinal()},

    'seawifs': {'INSTRUMENT': 'SeaWiFS',
                'PLATFORM': 'OrbView-2',
                'SEARCH': 'SeaWiFS_L2_',
                'SENSOR': '',
                'PERIOD_END': datetime(2010, 12, 11).toordinal(),
                'PERIOD_START': datetime(1997, 9, 4).toordinal()},

    'sgli': {'INSTRUMENT': 'SGLI',
             'PLATFORM': 'GCOM-C',
             'SEARCH': 'GC1SG1_*_L2SG_*_',
             'SENSOR': '',
             'PERIOD_END': datetime.today().toordinal(),
             'PERIOD_START': datetime(2018, 1, 1).toordinal()},

    'viirsn': {'INSTRUMENT': 'VIIRS',
               'PLATFORM': 'NPP',
               'SEARCH': 'VIIRSN_L2_',
               'SENSOR': 'vrsn',
               'PERIOD_END': datetime.today().toordinal(),
               'PERIOD_START': datetime(2012, 1, 2).toordinal()},

    'viirsj': {'INSTRUMENT': 'VIIRS',
               'PLATFORM': 'JPSS1',
               'SEARCH': 'VIIRSJ1_L2_',
               'SENSOR': '',
               'PERIOD_END': datetime.today().toordinal(),
               'PERIOD_START': datetime(2017, 11, 29).toordinal()}
}


def skip(mission: str, day: int):
    """
    Checks if sensor is within its lifespan
    :param mission:
    :param day:
    :return:
    """
    return (SATELLITES[mission]['PERIOD_START'] > day) or \
           (day > SATELLITES[mission]['PERIOD_END'])


class MatchUpError(Exception):
    """A custom exception used to report errors"""

    def __init__(self, message: str):
        super().__init__(message)


class UrlParser:
    def __init__(self, tim_min, tim_max, **kwargs):
        sat = kwargs.pop('sat')
        self.instrument = SATELLITES[sat]['INSTRUMENT']
        self.platform = SATELLITES[sat]['PLATFORM']
        self.sen = SATELLITES[sat]['SENSOR']
        self.tim_min = tim_min
        self.tim_max = tim_max
        self.slon = None
        self.slat = None
        self.elat = None
        self.elon = None
        self.data_type = ''
        self.sst_flag = None
        if kwargs['sst_flag'] is not None:
            self.sst_flag = kwargs['sst_flag'][0]
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.short_name = f"{SATELLITES[sat]['SEARCH']}{self.data_type.upper()}"
        if self.data_type.lower() == 'sst':
            platform, instrument = (f'S{self.platform}', self.instrument) \
                if sat == 'viirsn' else (self.platform, self.instrument)
            self.short_name = f'{platform}_{instrument}*.L2.SST.nc'
            if self.sst_flag in ('3', '4'):
                self.short_name = f'{platform}_{instrument}*.L2.SST{self.sst_flag}.nc'

    def cmr_point(self):
        if self.platform in ('JPSS1', 'ENVISAT'):
            return "https://cmr.earthdata.nasa.gov/search/granules.json?page_size=2000" \
                   "&sort_key=short_name" \
                   "&sort_key=start_date" \
                   "&provider=OB_DAAC" \
                   "&short_name={self.short_name}" \
                   f"&temporal={self.tim_min.strftime('%Y-%m-%dT%H:%M:%SZ')}," \
                   f"{self.tim_max.strftime('%Y-%m-%dT%H:%M:%SZ')}" \
                   f"&point={self.slon},{self.slat}" \
                   "&options[short_name][pattern]=true"

        if 'SST' in self.short_name:
            return self.ocbrowser()

        return "https://cmr.earthdata.nasa.gov/search/granules.json?page_size=2000" \
               "&sort_key=short_name" \
               "&sort_key=start_date" \
               "&provider=OB_DAAC" \
               f"&short_name={self.short_name}" \
               f"&temporal={self.tim_min.strftime('%Y-%m-%dT%H:%M:%SZ')}," \
               f"{self.tim_max.strftime('%Y-%m-%dT%H:%M:%SZ')}" \
               f"&point={self.slon},{self.slat}" \
               "&options[short_name][pattern]=true"

    def cmr_polygon(self):
        if self.platform in ('JPSS1', 'ENVISAT'):
            return "https://cmr.earthdata.nasa.gov/search/granules.json?page_size=2000" \
                   "&provider=OB_DAAC" \
                   f"&bounding_box={self.slon},{self.slat},{self.elon},{self.elat}" \
                   f"&instrument={self.instrument}" \
                   f"&platform={self.platform}" \
                   f"&short_name={self.short_name}" \
                   f"&temporal={self.tim_min.strftime('%Y-%m-%dT%H:%M:%SZ')}," \
                   f"{self.tim_max.strftime('%Y-%m-%dT%H:%M:%SZ')}" \
                   "&sort_key=short_name" \
                   "&options[short_name][pattern]=true"

        if 'SST' in self.short_name:
            return self.ocbrowser()

        return "https://cmr.earthdata.nasa.gov/search/granules.json?page_size=2000" \
               "&provider=OB_DAAC" \
               f"&bounding_box={self.slon},{self.slat},{self.elon},{self.elat}" \
               f"&instrument={self.instrument}" \
               f"&platform={self.platform}" \
               f"&short_name={self.short_name}" \
               f"&temporal={self.tim_min.strftime('%Y-%m-%dT%H:%M:%SZ')}," \
               f"{self.tim_max.strftime('%Y-%m-%dT%H:%M:%SZ')}&" \
               "sort_key=short_name" \
               "&options[short_name][pattern]=true"

    def csw_url(self):
        # return 'https://gportal.jaxa.jp/csw/csw?service=CSW&version=3.0.0' \
        #        '&request=GetRecords&outputFormat=application/json' \
        #        f'&datasetId={self.get_id()}&bbox={self.slon},{self.slat},{self.elon},{self.elat}' \
        #        f'&startTime={self.tim_min.strftime("%Y-%m-%dT%H:%M:%SZ")}' \
        #        f'&endTime={self.tim_max.strftime("%Y-%m-%dT%H:%M:%SZ")}&count=1000&pslv=L2'
        return 'https://gportal.jaxa.jp/csw/csw?service=CSW' \
               '&version=3.0.0' \
               '&request=GetRecords' \
               '&outputFormat=application/json' \
               '&pslv=L2' \
               '&count=2000' \
               '&sen=SGLI' \
               '&sat=GCOM-C' \
               f'&bbox={self.slon},{self.slat},{self.elon},{self.elat}' \
               f'&datasetId={self.get_id()}' \
               f'&startTime={self.tim_min.strftime("%Y-%m-%dT%H:%M:%SZ")}' \
               f'&endTime={self.tim_max.strftime("%Y-%m-%dT%H:%M:%SZ")}'

    def csw_polygon(self):
        return 'https://gportal.jaxa.jp/csw/csw?service=CSW' \
               '&version=3.0.0' \
               '&request=GetRecords' \
               '&outputFormat=application/json' \
               '&pslv=L2' \
               '&sat=GCOM-C' \
               '&count=2000' \
               '&sen=SGLI' \
               f'&bbox={self.slon},{self.slat},{self.elon},{self.elat}' \
               f'&datasetId={self.get_id()}' \
               f'&startTime={self.tim_min.strftime("%Y-%m-%dT%H:%M:%SZ")}' \
               f'&endTime={self.tim_max.strftime("%Y-%m-%dT%H:%M:%SZ")}'

    def get_id(self):
        """
        L2 Chlorophyll-a concentration etc.: 10002001
        L2 SST : 10002002 (Daytime and Nighttime are the same Dataset ID.)
        L2 NWLR : 10002000 (Including "Normalized water leaving radiance",
                            "Photosynthetically available radiation" and
                            "Atmosphere correction parameter".)
        """
        if self.data_type == '*':
            return '10002000,10002001,10002002'
        if self.data_type == 'rrs':
            return '10002000'
        if self.data_type == 'oc':
            return '10002001'
        if self.data_type == 'sst':
            return '10002002'
        if self.data_type == 'iop':
            raise MatchUpError('IOP not defined for SGLI')

    def ocbrowser(self):
        day = date2num(self.tim_min, calendar=u'gregorian',
                       units=u'days since 1970-01-01 00:00:00')
        dnm, prm = f'&dnm={self.sst_flag.upper()}', 'SST'
        if self.sst_flag in ('3', '4'):
            dnm, prm = '', f'SST{self.sst_flag}'

        return f'https://oceancolor.gsfc.nasa.gov/cgi/browse.pl?sub=level1or2list' \
               f'&sen={self.sen}' \
               '&per=DAY' \
               f'&day={day}' \
               f'&n={self.elat}' \
               f'&s={self.slat}' \
               f'&w={self.slon}' \
               f'&e={self.elon}' \
               f'{dnm}' \
               f'&prm={prm}'


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
        logger.info(f'{ds}\n{columns}\n{ds.shape}')

    if len(columns):
        ds.rename(columns=columns, inplace=True)

    date_time = None
    idx = [i for i, col in enumerate(ds.columns)
           if col.lower() in ['date', 'year', 'month', 'day']]
    idx += [i for i, col in enumerate(ds.columns)
            if col.lower() in ['time', 'hour', 'minute', 'second']]

    if not debug:
        logger.info(idx)

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
    if debug:
        logger.info(ds)
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


class FileSanity:
    def __init__(self, check_list: list, instrument: str, logger,
                 host: str = 'None', control_list: list = None):
        self.check_list = check_list
        self.instrument = instrument
        self.logger = logger
        self.host = host
        self.control_list = control_list
        if control_list is None:
            self.control_list = []

    def file_check(self, file: Path, sds=None) -> masked_array:

        basename = file.name
        if self.instrument in ('octs', 'seawifs', 'modisa', 'viirsn', 'viirsj', 'goci'):
            key = 'chlor_a' if 'OC' in basename else 'sst4' if 'SST4' in basename else 'sst'
            with Dataset(file, 'r') as dst:
                if 'IOP' in basename:
                    key = list(dst.groups['geophysical_data'].variables.keys())[0]
                sds = dst.groups['geophysical_data'][key][:]
            return sds

        # ~~~~~<>><<>><<>><<>><<>><<>><<>><~~~~~
        if self.instrument == 'meris':
            sds_obj = SD(file, SDC.READ)
            sds = sds_obj.select('chlor_a')
            fill_value = sds.bad_value_scaled
            sds = sds.get()  # select sds
            sds_obj.end()
            sds = np.ma.masked_where(np.equal(sds, fill_value), sds)
            return sds

        if self.instrument == 'sgli':
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

    def check(self) -> list:
        # Sometimes there are empty files that
        # need to be taken care of before mapping...
        keep_files = []
        append = keep_files.append
        cmd = 'del /f {file} >nul' if (
                os.name == 'nt') else 'rm -f {file} 2> /dev/null'

        # self.logger.info(f'check_list: {self.check_list}')
        for i, file in enumerate(self.check_list):
            check_file = Path(file)
            bsn = check_file.name
            if f'{bsn}:OK\n' in self.control_list:
                append(check_file)
                continue
            # self.logger.info(f'check_file: {check_file}')

            if not (bsn.endswith('.nc')
                    or bsn.endswith('.hdf')
                    or bsn.endswith('.h5')):
                continue

            try:
                data = self.file_check(file=check_file)
            except Exception as exc:
                if check_file.is_file():
                    subprocess.call(cmd.format(file=check_file.absolute()),
                                    shell=True)
                if self.logger:
                    self.logger.exception(f'\tFile#: {(i + 1): 3d} | {bsn} | {self.instrument}\n{exc}')
                if self.host == 'npec':
                    print(f'\tFile#: {(i + 1): 3d} | {bsn} | {self.instrument}\n{exc}', file=sys.stderr)
                continue

            if data is None:
                subprocess.call(cmd.format(file=check_file), shell=True)
                if self.logger:
                    self.logger.warning(f'\tFile#: {(i + 1): 3d} | {bsn}: BadFile, removed')
                if self.host == 'npec':
                    print(f'\tFile#: {(i + 1): 3d} | {bsn}: BadFile, removed', file=sys.stderr)
                continue

            if data[~data.mask].size == 0:
                subprocess.call(cmd.format(file=check_file), shell=True)
                if self.logger:
                    self.logger.warning(f'\tFile#: {(i + 1): 3d} | {bsn}: Empty, removed')
                if self.host == 'npec':
                    print(f'\tFile#: {(i + 1): 3d} | {bsn}: Empty, removed', file=sys.stderr)
                continue

            if data[~data.mask].size > 0:
                if self.logger:
                    self.logger.info(f'\tFile#: {(i + 1): 3d} | {bsn}: Pass')
                append(check_file)
        return keep_files
