#!/usr/bin/env python3
# coding: utf-8
"""
Name:        online file search/get
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
import os
import re
import subprocess
from netrc import netrc
from pathlib import Path
from pprint import pprint

import requests
from requests.adapters import HTTPAdapter


def get_auth(host: str):
    """
    Retrieve my credentials for satellite data download
    @param host:
    @return:
    """
    net_rc = netrc()
    user, _, passwd = net_rc.authenticators(host)
    return user, passwd


def obpg_search(query: str):
    with requests.Session() as request:
        request.mount('https://', HTTPAdapter(max_retries=3))
        resp = request.get(query, timeout=30)
    return get_filename_list(response=resp, query=query)


def get_filename_list(response: requests, query: str) -> list:
    url = query[:-len(Path(query).name) - 1]
    get_file = 'https://oceandata.sci.gsfc.nasa.gov/ob/getfile'

    regex = re.compile(r'filenamelist&id=(\d+\.\d+)')
    fid = regex.findall(response.text)
    prm = query[query.index('&prm'):].split('=')[1]

    if len(fid) == 0:
        start = query.index('&sen=')
        end = query.index('&per=')
        sen = {'amod': 'AQUA_MODIS',
               'tmod': 'TERRA_MODIS',
               'vrsn': 'SNPP_VIIRS',
               }[query[start:end].split('=')[1]]

        files = re.findall(f'file=(.*?{sen}.*L2.{prm}.nc)', response.text)
        if len(files) > 0:
            return [f'{get_file}/{f}' for f in files if '&' not in f]

    if len(fid) == 0:
        print(f'Download >> {query} >> No Files Found...!!! ')
        return []

    response = requests.get(f'{url}/browse.pl?sub=filenamelist&id='
                            f'{fid[0]}&prm={prm}')
    if response.reason == 'Not Found':
        return []

    return [f'{get_file}/{f}'
            for f in response.text.splitlines()]


def fmt_content(files: list, sst_flag: str = None):
    contents = {'feed': {'entry': []}}
    append = contents['feed']['entry'].append

    for href in files:
        producer_granule_id = Path(href).name
        if sst_flag and (sst_flag not in producer_granule_id):
            continue
        entry = {'producer_granule_id': producer_granule_id,
                 'links': [{'href': href}]}
        append(entry)
    return contents


def search(url: str, sen: str, debug, sst_flag: str = None):
    """ function to submit a given URL request to the CMR; return JSON output """

    if (sen != 'sgli') and ('SST' in url):
        files = obpg_search(query=url)
        return fmt_content(files=list(set(files)))

    response = requests.get(url)

    if sen != 'sgli':
        content = response.json()
        if debug:
            pprint(f'{content}\n{url}')
        return content

    if response.status_code != 200:
        return []
    content = response.json()
    if debug:
        pprint(f'{content}\n{url}')

    if content['properties']['numberOfRecordsReturned'] == 0:
        return []
    files = re.findall('standard/GCOM-C/GCOM-C.SGLI/'
                       'L2.OCEAN.*/GC1SG1_.*Q_.*.h5',
                       '\n'.join([feature['properties']['product']['fileName']
                                  for feature in content['features']]))
    sst_flag = f'SST{sst_flag}' if sst_flag else sst_flag
    if len(files) == 0:
        return []
    return fmt_content(files=files, sst_flag=sst_flag)


def wget(url: str, out_dir: Path, case: str, logger):
    """
    cmr_download_file downloads a file
    given URL and out_dir strings
    syntax fname_local = cmr_download_file(url, out_dir)
    """

    bsn = Path(url).name
    local_dir = out_dir.absolute()
    local_filename = local_dir.joinpath(bsn)
    control_file = local_dir.joinpath('control_list.txt')

    if control_file.is_file():
        with open(control_file, 'r') as txt:
            file_list = list(filter(None, txt.readlines()))
            in_list = f'{local_filename.name}:OK\n' in file_list

        if in_list:
            logger.info(f'{local_filename}\nSUCCESS...! Downloaded file\n')
            return local_filename

    if case == 'cmr':
        host = 'urs.earthdata.nasa.gov'
        user, passwd = get_auth(host=host)
        cookies = Path().home().joinpath('.urs_cookies')
        cmd = 'wget -nc ' \
              '--auth-no-challenge=on ' \
              '--keep-session-cookies ' \
              f'--load-cookies {cookies} ' \
              f'--save-cookies {cookies} ' \
              f'--content-disposition "{url}" ' \
              f'--directory-prefix={local_dir}'
        if os.name == 'nt':
            cmd = f'{cmd} ' \
                  f'--password={passwd} ' \
                  f'--user={user}'
        # logger.info(cmd)
        status = subprocess.check_call(cmd, shell=True)
        if status != 0:
            logger.warning(f'WGET exit status: {status}')

        if status == 0:
            logger.info('SUCCESS!')
        return local_filename

    if case == 'csw':
        """
        Data retrieval function
        @return: a list of downloaded files
        """
        # get auth
        host = 'ftp.gportal.jaxa.jp'
        user, passwd = get_auth(host=host)

        cmd = 'wget -nc ' \
              '--preserve-permissions ' \
              '--remove-listing ' \
              '--tries=5 ' \
              f'ftp://{user}:{passwd}@{host}/{url} ' \
              f'--directory-prefix={local_dir}'
        status = subprocess.check_call(cmd, shell=True)
        # status = os.system(cmd)

        # -C, --continue-at <offset>
        #   Continue/Resume a previous file transfer at the given offset.
        #   Use "-C -" to tell curl to automatically find out where/how to resume the transfer.
        #   It then uses the given output/input files to figure that out.
        #
        # -L, --location
        #   This option will make curl redo the request on the new place if the server reports
        #   that the requested page has moved to a different location (indicated with a Location:
        #   header and a 3XX response code),

        # cmd = f'curl -L -C - ftp://{user}:{passwd}@{host}/{url} --output {bsn}'
        # status = subprocess.check_call(cmd, cwd=local_dir, shell=True)
        if status != 0:
            logger.warning(f'CMD: {cmd}\nStatus: {status}')
        if status == 0:
            logger.info('SUCCESS!')
        return local_filename


def getfile(content, out_dir: Path, case: str, logger):
    """ function to process the return from a single CMR JSON return """

    download_files = []
    append = download_files.append

    for entry in content['feed']['entry']:
        granid = entry['producer_granule_id']

        if 'SST.NRT.nc' in granid:
            this_f = entry['links'][0]['href']
            print(f'Download\n\tID: {granid}\n\tLink: {this_f} | Skipping...\n')
            continue

        if out_dir:
            local_filename = wget(url=entry['links'][0]['href'],
                                  out_dir=out_dir,
                                  case=case,
                                  logger=logger)
            append(local_filename)
        else:
            append(entry['links'][0]['href'])

    # logger.info(download_files)
    return download_files
