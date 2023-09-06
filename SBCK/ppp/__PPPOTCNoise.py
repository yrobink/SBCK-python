
## Copyright(c) 2022 Yoann Robin
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

###############
## Libraries ##
###############

import numpy as np
from .__PrePostProcessing import PrePostProcessing


###########
## Class ##
###########

class PPPOTCNoise(PrePostProcessing):
	"""
	SBCK.ppp.PPPOTCNoise
	====================
	
	Add a random uniform noise in the cells defined by bin_width.
	
	"""
	
	def __init__( self , *args , **kwargs ):
		PrePostProcessing.__init__( self , *args , **kwargs )
	
	def transform( self , X ):
		return X
	
	def itransform( self , Xt ):
		bw = np.array( self._bc_method.bin_width).ravel()
		noise = np.random.uniform( low = -bw / 2 , high = bw / 2 , size = Xt.shape )
		return Xt + noise


