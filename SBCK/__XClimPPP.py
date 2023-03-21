
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

class XClimSPPP(PrePostProcessing):
	"""
	SBCK.XClimSPPP
	==============
	
	Experimental: just a class based on SBCK.ppp.PrePostProcessing for xclim,
	stationary case
	
	
	"""
	def __init__( self , **kwargs ):
		"""
		Initialisation of XClimSPPP.
		
		kwargs are directly given to SBCK.ppp.PrePostProcessing, only keywords
		arguments are available.
		
		"""
		PrePostProcessing.__init__( self , **kwargs )
	
	def fit( self , Y0 , X0 , X1 = None ):
		PrePostProcessing.fit( self , Y0 = Y0 , X0 = X0 )
	
	def predict( self , X1 , X0 = None ):
		return PrePostProcessing.predict( self , X1 )


class XClimNPPP(PrePostProcessing):
	"""
	SBCK.XClimNPPP
	==============
	
	Experimental: just a class based on SBCK.ppp.PrePostProcessing for xclim,
	non-stationary case
	
	
	"""
	def __init__( self , **kwargs ):
		"""
		Initialisation of XClimNPPP.
		
		kwargs are directly given to SBCK.ppp.PrePostProcessing, only keywords
		arguments are available.
		
		"""
		PrePostProcessing.__init__( self , **kwargs )
	
	def fit( self , Y0 , X0 , X1 ):
		PrePostProcessing.fit( self , Y0 = Y0 , X0 = X0 , X1 = X1 )
	
	def predict( self , X1 , X0 = None ):
		return PrePostProcessing.predict( self , X1 = X1 , X0 = X0 )
