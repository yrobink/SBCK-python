# -*- coding: utf-8 -*-

## Copyright(c) 2021 / 2024 Yoann Robin
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


from .__linalg    import as2d
from .__linalg    import sqrtm
from .__linalg    import choleskym

from .__stats     import bin_width_estimator
from .__stats     import rvs_spd_matrix

from .__SparseHist import SparseHist

from .__misc      import SlopeStoppingCriteria
from .__misc      import Shift
from .__misc      import yearly_window

from .__OT        import POTemd
from .__OT        import OTSinkhorn
from .__OT        import OTSinkhornLogDual

from .__shuffle   import schaake_shuffle
from .__shuffle   import SchaakeShuffle
from .__shuffle   import SchaakeShuffleRef
from .__shuffle   import MVQuantilesShuffle
from .__shuffle   import MVRanksShuffle

from .__rv_extend import rv_base
from .__rv_extend import rv_empirical
from .__rv_extend import rv_empirical_ratio
from .__rv_extend import rv_empirical_gpd
from .__rv_extend import rv_density
from .__rv_extend import rv_mixture
from .__rv_extend import mrv_base

from .__rv_extend import rv_histogram
from .__rv_extend import rv_ratio_histogram


