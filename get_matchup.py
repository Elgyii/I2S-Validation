import codecs
import subprocess
from pathlib import Path

if __name__ == '__main__':

    BASEDIR = Path().absolute()  # Base data directory
    mf = BASEDIR.joinpath('matchup.txt')
    with codecs.open(str(mf), "r", "utf-8") as txt:
        match_cmd = [line.strip('\n')
                     for line in txt.readlines()
                     if (len(line) > len('\n')) or ('cd C:' not in line)]

    for cmd in match_cmd:
        if not ('goci' in cmd):
            continue
        if ('china' in cmd) or \
                ('korea' in cmd):

            continue
        print(cmd)
        subprocess.check_call(cmd, shell=True)
