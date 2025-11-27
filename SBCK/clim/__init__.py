
## Copyright(c) 2025 Yoann Robin
## 
## This file is part of SBCK.
## 
## SBCK is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## SBCK is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with SBCK.  If not, see <https://www.gnu.org/licenses/>.

from .__tools import yearly_window
from .__apply_bcm import apply_bcm

from .__stats import phaversine_distances
from .__stats import xcorr
from .__stats import cacorrelogram
from .__stats import cadescribe

## Import only if zxarray is available
import importlib.util
if importlib.util.find_spec("zxarray"):
    from .__zstats import zcorr
    from .__zstats import zcacorrelogram
    from .__zstats import zcadescribe
