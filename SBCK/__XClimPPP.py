
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

from .ppp.__PrePostProcessing import PrePostProcessing


###########
## Class ##
###########

class XClimPPP(PrePostProcessing):
	"""
	SBCK.XClimPPP
	=============
	
	Experimental: just a class based on SBCK.ppp.PrePostProcessing for xclim
	
	
	"""
	def __init__( self , **kwargs ):
		"""
		Initialisation of XClimPPP.
		
		kwargs are directly given to SBCK.ppp.PrePostProcessing, only keywords
		arguments are available.
		
		"""
		PrePostProcessing.__init__( self , **kwargs )


