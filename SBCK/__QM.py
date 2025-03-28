# -*- coding: utf-8 -*-

## Copyright(c) 2021 / 2025 Yoann Robin
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

from .__AbstractBC import UnivariateBC
from .__AbstractBC import MultiUBC
from .tools.__rv_extend import WrapperStatisticalDistribution
from .tools.__rv_extend import rv_empirical


###########
## Class ##
###########

class Univariate_QM(UnivariateBC):##{{{
	
	def __init__( self , rvY = rv_empirical , rvX = rv_empirical ):
		super().__init__( "Univariate_QM" , "S" )
		self._rvY = rvY
		self._rvX = rvX
		self.rvY0 = WrapperStatisticalDistribution(self._rvY)
		self.rvX0 = WrapperStatisticalDistribution(self._rvX)
	
	def fit( self , Y0 , X0 ):
		self.rvX0.fit(X0)
		self.rvY0.fit(Y0)
		
		return self
	
	def _predictZ0( self , X0 , reinfer_X0 = False , **kwargs ):
		if X0 is None:
			return None
		cdfX0 = self.rvX0.cdf
		if reinfer_X0:
			rvX0 = WrapperStatisticalDistribution(self._rvX)
			rvX0.fit(X0)
			cdfX0 = rvX0.cdf
		eps  = np.sqrt(np.finfo(X0.dtype).resolution)
		cdf  = cdfX0(X0)
		cdfx = max( 1 - ( 1 - cdf[cdf < 1].max() / 10 ) , 1 - eps )
		cdfn = min(           cdf[cdf > 0].min() / 10   ,     eps )
		cdf = np.where( cdf < 1 , cdf , cdfx )
		cdf = np.where( cdf > 0 , cdf , cdfn )
		return self.rvY0.icdf(cdf)
##}}}

class QM(MultiUBC):##{{{
	
	"""
	SBCK.QM
	=======
	Quantile Mapping bias corrector, see e.g. [1,2,3]. The implementation
	proposed here is generic, and can use scipy.stats to fit a parametric
	distribution, or can use a frozen distribution.
	
	Example
	-------
	```
	## Start with imports
	from SBCK import QM
	from SBCK.tools import rv_empirical
	
	## Start by define two kinds of laws from scipy.stats, the Normal and Exponential distribution
	norm  = sc.norm
	expon = sc.expon
	
	## And define calibration and projection dataset such that the law of each columns are reversed
	size = 10000
	Y0   = np.stack( [expon.rvs( scale = 2 , size = size ),norm.rvs( loc = 0 , scale = 1 , size = size )] ).T
	X0   = np.stack( [norm.rvs( loc = 0 , scale = 1 , size = size ),expon.rvs( scale = 1 , size = size )] ).T
	
	## Generally, the law of Y0 and X0 is unknow, so we use the empirical histogram distribution
	qm   = bc.QM( rvY = [bct.rv_empirical,bct.rv_empirical] , rvX = [bct.rv_empirical,bct.rv_empirical] ).fit( Y0 , X0 )
	Z0_h = qm.predict(X0)
	
	## Actually, this is the default behavior
	qm = bc.QM().fit( Y0 , X0 )
	assert np.abs(Z0_h - qm.predict(X0)).max() < 1e-12
	
	## In some case we know the kind of law of Y0 (or X0)
	qm    = bc.QM( rvY = [expon,norm] ).fit( Y0 , X0 )
	Z0_Y0 = qm.predict(X0)
	
	## Or, even better, we know the law of the 2nd component of Y0 (or X0)
	qm    = bc.QM( rvY = [expon,norm(loc=0,scale=1)] ).fit( Y0 , X0 )
	Z0_Y2 = qm.predict(X0)
	
	## Obviously, we can mix all this strategy to build a custom Quantile Mapping
	qm    = bc.QM( rvY = [bct.rv_empirical,norm(loc=0,scale=1)] , rvX = [norm,bct.rv_empirical] ).fit( Y0 , X0 )
	Z0_Yh = qm.predict(X0)
	```
	
	References
	----------
	[1] Panofsky, H. A. and Brier, G. W.: Some applications of statistics to
	meteorology, Mineral Industries Extension Services, College of Mineral
	Industries, Pennsylvania State University, 103 pp., 1958.
	[2] Wood, A. W., Leung, L. R., Sridhar, V., and Lettenmaier, D. P.:
	Hydrologic Implications of Dynamical and Statistical Approaches to
	Downscaling Climate Model Outputs, Clim. Change, 62, 189–216,
	https://doi.org/10.1023/B:CLIM.0000013685.99609.9e, 2004.
	[3] Déqué, M.: Frequency of precipitation and temperature extremes over
	France in an anthropogenic scenario: Model results and statistical
	correction according to observed values, Global Planet. Change, 57, 16–26,
	https://doi.org/10.1016/j.gloplacha.2006.11.030, 2007.
	"""
	
	def __init__( self , rvY = rv_empirical , rvX = rv_empirical ):##{{{
		"""
		SBCK.QM.__init__
		================
		
		Arguments
		---------
		rvY: SBCK.tools.<law> | scipy.stats.<law>
			Law of references
		rvX: SBCK.tools.<law> | scipy.stats.<law>
			Law of models
		"""
		
		## Build args for MultiUBC
		if not isinstance( rvY , (list,tuple) ):
			if isinstance( rvX , (list,tuple) ):
				rvY = [rvY for _ in range(len(rvX))]
			else:
				rvY = [rvY]
		if not isinstance( rvX , (list,tuple) ):
			if isinstance( rvY , (list,tuple) ):
				rvX = [rvX for _ in range(len(rvY))]
			else:
				rvX = [rvX]
		if not len(rvX) == len(rvY):
			raise ValueError( "Incoherent arguments between rvY and rvX" )
		args = [ (rvy,rvx) for rvy,rvx in zip(rvY,rvX) ]
		
		## And init upper class
		super().__init__( "QM" , Univariate_QM , args = args )
	##}}}
	
##}}}


