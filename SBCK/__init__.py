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

from .__release import version
__version__ = version

## Import sub modules
from . import clim
from . import datasets
from . import misc
from . import mm
from . import ppp
from . import stats

## Import main classes
from .__AbstractBC import AbstractBC

from .__miscBC import RBC
from .__miscBC import IdBC

from .__QM     import QM
from .__CDFt   import CDFt

from .__R2D2 import  R2D2
from .__R2D2 import AR2D2
from .__R2D2 import  QMrs

from .__dOTC import  OTC
from .__dOTC import dOTC
from .__dOTC import dOTC1d

from .__dTSMBC import  TSMBC
from .__dTSMBC import dTSMBC

from .__QDM import QDM
from .__QQD import QQD

from .__others_Nd import MBCn
from .__others_Nd import MRec
from .__others_Nd import XClimNPPP
from .__others_Nd import XClimSPPP

