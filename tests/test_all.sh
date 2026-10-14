#!bin/bash

python3 test_SBCK.py $1
python3 test_SBCK_stats.py $1
python3 test_SBCK_ppp.py $1
python3 test_SBCK_clim.py $1
python3 test_SBCK_mm.py $1
