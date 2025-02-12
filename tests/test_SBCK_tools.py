#!/usr/bin/env python3 -m unittest

## Copyright(c) 2024 Yoann Robin
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


#############
## Imports ##
#############

import os
import itertools as itt
import unittest

import numpy as np
import scipy.stats as sc

import SBCK as bc
import SBCK.tools as bct
import SBCK.datasets as bcd
import SBCK.metrics as bcm

import matplotlib as mpl
import matplotlib.pyplot   as plt
import matplotlib.gridspec as mplg


########################
## Set mpl parameters ##
########################

mpl.rcdefaults()
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.linewidth']  = 0.5
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['patch.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5


######################
## Parameters class ##
######################

class SBCKTestParameters:##{{{
	
	def __init__( self ):##{{{
		lpath = os.path.join( *os.path.basename(__file__).split(".")[0].split("_")[1:] )
		self.opath = os.path.join( os.path.dirname(__file__) , "figures" , lpath )
		if not os.path.isdir(self.opath):
			os.makedirs(self.opath)
	##}}}
	
	@property
	def prefix(self):
		return str(type(self)).split("'")[1].split(".")[-1]
	
##}}}


############################
## SBCK.tools.__rv_extend ##
############################

class RVExtend(SBCKTestParameters):##{{{
	
	def __init__( self , tex_name , rv_class , data_type = "tas-pr" , **kwargs ):##{{{
		super().__init__()
		self.tex_name  = tex_name
		self.rv_class  = rv_class
		self.rv_kwargs = kwargs
		self.data_type = data_type
	##}}}
	
	def test_rv_methods(self):##{{{
		
		## Data
		size = 10_000
		e    = 0.01
		np.random.seed(42)
		if self.data_type == "tas-pr":
			Y0,_,_ = bc.datasets.like_tas_pr(size)
			xlim = [ [(-2,2  ),(-e,1+e),(-2,2  )] , [(-0.5,7  ),(-e  ,1+e),(-0.5,7  )] ]
			ylim = [ [(-e,1+e),(-2,2  ),( 0,1.3)] , [(-e  ,1+e),(-0.5,7  ),( 0  ,1.1)] ]
		elif self.data_type == "L":
			Y0,_,_ = bc.datasets.gaussian_L_2d(size)
			xlim = [ [(-5,6  ),(-e,1+e),(-5,6  )] , [(-5,7  ),(-e,1+e),(-5,7  )] ]
			ylim = [ [(-e,1+e),(-5,6  ),( 0,0.5)] , [(-e,1+e),(-5,7  ),( 0,0.5)] ]
		
		## Figure
		fig  = plt.figure()
		grid = mplg.GridSpec( 2 * 2 + 1 , 2 * 3 + 1 )
		
		ax = fig.add_subplot(grid[0,1:-1])
		ax.text( 0 , 0 , self.prefix , ha = "center" , va = "center" , fontdict = { "weight" : "bold" , "size" : 11 } )
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		ax.set_axis_off()
		
		## Loop
		for i in range(2):
			
			X = Y0[:,i]
			
			## Random variable
			rvX = self.rv_class( *self.rv_class.fit(X) )
			
			## Draw and re-init
			R   = rvX.rvs(size)
			rvR = self.rv_class( *self.rv_class.fit(R) )
			
			## Data
			dx   = 0.05 * (X.max() - X.min())
			xmin = X.min() - dx
			xmax = X.max() + dx
			x    = np.linspace( xmin , xmax , 1000 )
			bins = np.linspace( xmin , xmax , 51 )
			p    = np.linspace( 0 , 1 , 1000 )
			
			##
			ax = fig.add_subplot(grid[2*i+1,1])
			ax.plot( x , rvX.cdf(x) , color = "red"  , label = "CDF" )
			ax.plot( x , rvX.sf(x)  , color = "blue" , label = "SF"  )
			ax.legend( loc = "upper left" )
			ax.set_xlabel( r"$x$" )
			ax.set_ylabel( r"$p$" )
			ax.set_xlim(xlim[i][0])
			ax.set_ylim(ylim[i][0])
			
			ax = fig.add_subplot(grid[2*i+1,3])
			ax.plot( p , rvX.icdf(p) , color = "red"  , label = "CDF$^{-1}$" )
			ax.plot( p , rvX.isf(p)  , color = "blue" , label = "SF$^{-1}$"  )
			ax.legend( loc = "upper left" )
			ax.set_xlabel( r"$p$" )
			ax.set_ylabel( r"$x$" )
			ax.set_xlim(xlim[i][1])
			ax.set_ylim(ylim[i][1])
			
			ax = fig.add_subplot(grid[2*i+1,5])
			ax.plot( x , rvX.pdf(x) , color = "red" )
			ax.hist( X , bins , color = "red" , alpha = 0.5 , density = True )
			ax.plot( x , rvR.pdf(x) , color = "blue" , alpha = 0.5 )
			ax.hist( R , bins , color = "blue" , alpha = 0.2 , density = True )
			ax.set_xlabel( r"$x$" )
			ax.set_ylabel( r"Density" )
			ax.set_xlim(xlim[i][2])
			ax.set_ylim(ylim[i][2])
		
		## Figsize
		mm = 1. / 25.4
		pt = 1. / 72
		
		width  = 210*mm
		w_l    = 30*pt
		w_m    = 35*pt
		w_r    =  5*pt
		w_ax   = (width - (w_l + 2 * w_m + w_r)) / 3
		widths = [w_l,w_ax,w_m,w_ax,w_m,w_ax,w_r]
		
		h_ax    = w_ax / (16 / 11)
		h_t     = 20*pt
		h_m     = 30*pt
		h_b     = 30*pt
		heights = [h_t,h_ax,h_m,h_ax,h_b]
		height  = sum(heights)
		
		grid.set_height_ratios(heights)
		grid.set_width_ratios(widths)
		fig.set_figheight(height)
		fig.set_figwidth(width)
		
		plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
		plt.savefig( os.path.join( self.opath , f"{self.prefix}_rv_method.png" ) , dpi = 600 )
		plt.close(fig)
		
	##}}}
	
##}}}

class Test_rv_empirical(RVExtend,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		RVExtend.__init__( self , r"rv_empirical" , bc.tools.rv_empirical , "L" )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
##}}}

class Test_rv_empirical_ratio(RVExtend,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		RVExtend.__init__( self , r"rv_empirical_ratio" , bc.tools.rv_empirical_ratio , "tas-pr" )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
##}}}

class Test_rv_empirical_gpd(RVExtend,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		RVExtend.__init__( self , r"rv_empirical_gpd" , bc.tools.rv_empirical_gpd , "L" )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
##}}}

class Test_rv_density(RVExtend,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		RVExtend.__init__( self , r"rv_density" , bc.tools.rv_density , "L" )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
##}}}

class Test_rv_mixture(RVExtend,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		RVExtend.__init__( self , r"rv_mixture" , bc.tools.rv_mixture , "mixture" )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
	def test_rv_methods(self):##{{{
		
		## Data
		size = 10_000
		e    = 0.01
		np.random.seed(42)
		lrvX = [
				bct.rv_mixture( [0.75,0.25]   , sc.expon(scale = 1) , sc.norm(loc = 10,scale = 1) ),
				bct.rv_mixture( [0.1,0.4,0.5] , sc.norm( loc = -5 , scale = 1 ) , sc.norm( loc = 0 , scale = 0.5 ) , sc.norm( loc = 5 , scale = 2 ) )
		]
		xlim    = [ [(-0.5,7  ),(-e  ,1+e),(-0.5,7  )] , [(-9,12  ),(-e,1+e),(-9,12  )] ]
		ylim    = [ [(-e  ,1+e),(-0.5,7  ),( 0  ,0.8)] , [(-e  ,1+e),(-9,12  ),( 0  ,0.4)] ]
		
		## Figure
		fig  = plt.figure()
		grid = mplg.GridSpec( 2 * 2 + 1 , 2 * 3 + 1 )
		
		ax = fig.add_subplot(grid[0,1:-1])
		ax.text( 0 , 0 , self.prefix , ha = "center" , va = "center" , fontdict = { "weight" : "bold" , "size" : 11 } )
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		ax.set_axis_off()
		
		## Loop
		args = [ (sc.expon,sc.norm),
		         (sc.norm,sc.norm,sc.norm)]
		
		for i in range(2):
			
			## Random variable
			rvX = lrvX[i]
			X   = rvX.rvs(size)
			
			## Data
			dx   = 0.05 * (X.max() - X.min())
			xmin = X.min() - dx
			xmax = X.max() + dx
			x    = np.linspace( xmin , xmax , 1000 )
			bins = np.linspace( xmin , xmax , 51 )
			p    = np.linspace( 0 , 1 , 1000 )
			
			##
			ax = fig.add_subplot(grid[2*i+1,1])
			ax.plot( x , rvX.cdf(x) , color = "red"  , label = "CDF" )
			ax.plot( x , rvX.sf(x)  , color = "blue" , label = "SF"  )
			ax.legend( loc = "upper left" )
			ax.set_xlabel( r"$x$" )
			ax.set_ylabel( r"$p$" )
			ax.set_xlim(xlim[i][0])
			ax.set_ylim(ylim[i][0])
			
			ax = fig.add_subplot(grid[2*i+1,3])
			ax.plot( p , rvX.icdf(p) , color = "red"  , label = "CDF$^{-1}$" )
			ax.plot( p , rvX.isf(p)  , color = "blue" , label = "SF$^{-1}$"  )
			ax.legend( loc = "upper left" )
			ax.set_xlabel( r"$p$" )
			ax.set_ylabel( r"$x$" )
			ax.set_xlim(xlim[i][1])
			ax.set_ylim(ylim[i][1])
			
			ax = fig.add_subplot(grid[2*i+1,5])
			ax.plot( x , rvX.pdf(x) , color = "red" )
			ax.hist( X , bins , color = "red" , alpha = 0.5 , density = True )
			ax.set_xlabel( r"$x$" )
			ax.set_ylabel( r"Density" )
			ax.set_xlim(xlim[i][2])
			ax.set_ylim(ylim[i][2])
		
		## Figsize
		mm = 1. / 25.4
		pt = 1. / 72
		
		width  = 210*mm
		w_l    = 30*pt
		w_m    = 35*pt
		w_r    =  5*pt
		w_ax   = (width - (w_l + 2 * w_m + w_r)) / 3
		widths = [w_l,w_ax,w_m,w_ax,w_m,w_ax,w_r]
		
		h_ax    = w_ax / (16 / 11)
		h_t     = 20*pt
		h_m     = 30*pt
		h_b     = 30*pt
		heights = [h_t,h_ax,h_m,h_ax,h_b]
		height  = sum(heights)
		
		grid.set_height_ratios(heights)
		grid.set_width_ratios(widths)
		fig.set_figheight(height)
		fig.set_figwidth(width)
		
		plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
		plt.savefig( os.path.join( self.opath , f"{self.prefix}_rv_method.png" ) , dpi = 600 )
		plt.close(fig)
		
	##}}}
	
##}}}


#############################
## SBCK.tools.__SparseHist ##
#############################

class Test_SparseHist(SBCKTestParameters,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		SBCKTestParameters.__init__( self )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
	def test_mnormal(self):##{{{
		np.random.seed(42)
		X = np.random.multivariate_normal( mean = np.zeros(2) , cov = np.identity(2) , size = 1_000_000 )
		rvX = bct.SparseHist( X )
		
		grid = mplg.GridSpec(3,3)
		fig  = plt.figure()
		ax   = fig.add_subplot( grid[1,1] , projection = '3d' )
		ax.scatter( rvX.c[:,0] , rvX.c[:,1] , rvX.p , c = rvX.p , cmap = plt.cm.hot , linestyle = "" , marker = "." )
		ax.set_xlabel( r"$x$" )
		ax.set_ylabel( r"$y$" )
		ax.set_zlabel( r"$z$" )
		ax.set_xlim(-5,5)
		ax.set_ylim(-5,5)
		
		mm   = 1. / 25.4
		pt   = 1. / 72
		width  = 120*mm
		w_l    = 1*pt
		w_r    = 26*pt
		w_ax   = width - w_l - w_r
		widths = [w_l,w_ax,w_r]
		
		h_ax    = w_ax
		h_t     = 1*pt
		h_b     = 1*pt
		heights = [h_t,h_ax,h_b]
		height  = sum(heights)
		
		grid.set_height_ratios(heights)
		grid.set_width_ratios(widths)
		fig.set_figheight(height)
		fig.set_figwidth(width)
		
		plt.subplots_adjust( left = 0 , right = 1 , bottom = 0 , top = 1 , hspace = 0 , wspace = 0 )
		plt.savefig( os.path.join( self.opath , f"{self.prefix}_mnormal.png" ) , dpi = 600 )
	##}}}
	
##}}}

##########
## main ##
##########

if __name__ == "__main__":
	unittest.main()
