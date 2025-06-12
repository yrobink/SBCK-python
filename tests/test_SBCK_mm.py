#!/usr/bin/env python3 -m unittest

## Copyright(c) 2024, 2025 Yoann Robin
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

import sys
import os
import argparse
import itertools as itt
import unittest

import numpy as np
import scipy.stats as sc
import xarray as xr
import pandas as pd

import SBCK as bc
import SBCK.ppp as bcp
import SBCK.stats as bcs
import SBCK.datasets as bcd
import SBCK.mm as bcmm


has_mpl = True
try:
	import matplotlib as mpl
	import matplotlib.pyplot   as plt
	import matplotlib.gridspec as mplg
except Exception:
	has_mpl = False


########################
## Set mpl parameters ##
########################

try:
	mpl.rcdefaults()
	mpl.rcParams['font.size'] = 7
	mpl.rcParams['axes.linewidth']  = 0.5
	mpl.rcParams['lines.linewidth'] = 0.5
	mpl.rcParams['patch.linewidth'] = 0.5
	mpl.rcParams['xtick.major.width'] = 0.5
	mpl.rcParams['ytick.major.width'] = 0.5
except Exception:
	has_mpl = False


######################
## Parameters class ##
######################

class SBCKTestParameters:##{{{
	
	PLOTFIG: bool = False
	
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


#######################
## Multi-Model Tests ##
#######################

class Test_AlphaPooling(SBCKTestParameters,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		SBCKTestParameters.__init__( self )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
	def test_alphaPooling(self):##{{{
		
		## Parameters
		np.random.seed(42)
		size    = 1_000
		nmod    = 3
		locs0   = np.linspace( -5 , 5 , nmod )
		locs1   = np.linspace( 5 , 15 , nmod )
		scales1 = np.linspace( 0.1 , 3 , nmod )
		scales0 = np.linspace( 0.5 , 5 , nmod )
		
		## Data
		Y0  =  np.random.normal( loc = 0 , scale = 1 , size = size )
		X0s = [np.random.normal( loc = locs0[i] , scale = scales0[i] , size = size ) for i in range(nmod)]
		X1s = [np.random.normal( loc = locs1[i] , scale = scales1[i] , size = size ) for i in range(nmod)]
		
		## mm correction
		mm  = bcmm.AlphaPooling( alpha = 1 ).fit( Y0 , X0s , X1s )
		Z1s,Z0s  = mm.predict( X1s , X0s )
		
		## Random variables
		rvY0  = bcs.rv_empirical.fit( Y0 )
		rvX0s = [ bcs.rv_empirical.fit( X0 ) for X0 in X0s ]
		rvX1s = [ bcs.rv_empirical.fit( X1 ) for X1 in X1s ]
		rvZ0s = [ bcs.rv_empirical.fit( Z0 ) for Z0 in Z0s ]
		rvZ1s = [ bcs.rv_empirical.fit( Z1 ) for Z1 in Z1s ]
		xmin  = min( [rv.a for rv in [rvY0] + rvX0s + rvX1s + rvZ0s + rvZ1s] )
		xmax  = max( [rv.b for rv in [rvY0] + rvX0s + rvX1s + rvZ0s + rvZ1s] )
		dx    = 0.05 * (xmax - xmin)
		x     = np.linspace( xmin - dx , xmax + dx , 1000 )
		
		## Plot
		if self.PLOTFIG:
			fig  = plt.figure()
			grid = mplg.GridSpec( 2 * 2 + 1 , 2 * 2 + 1 )
			
			ax  = fig.add_subplot(grid[1,1])
			ax.plot( x , rvY0.cdf(x) , color = "blue" , label = r"$Y^0$" )
			for rvX0 in rvX0s:
				ax.plot( x , rvX0.cdf(x) , color = "red" , label = r"$X^0$" )
			ax.spines[['right', 'top']].set_visible(False)
			ax.legend( loc = "lower right" )
			ax.set_xticks([])
			ax.set_ylabel("CDF")
			
			ax  = fig.add_subplot(grid[3,1])
			ax.plot( x , rvY0.cdf(x) , color = "blue" , label = r"$Y^0$" )
			for rvZ0 in rvZ0s:
				ax.plot( x , rvZ0.cdf(x) , color = "green" , label = r"$Z^0$" )
			ax.spines[['right', 'top']].set_visible(False)
			ax.legend( loc = "lower right" )
			ax.set_xlabel(r"$x$")
			ax.set_ylabel("CDF")
			
			ax  = fig.add_subplot(grid[1,3])
			ax.plot( x , rvY0.cdf(x) , color = "blue" , label = r"$Y^0$" )
			for rvX1 in rvX1s:
				ax.plot( x , rvX1.cdf(x) , color = "purple" , label = r"$X^1$" )
			ax.spines[['right', 'top']].set_visible(False)
			ax.legend( loc = "lower right" )
			ax.set_xticks([])
			ax.set_yticks([])
			
			ax  = fig.add_subplot(grid[3,3])
			ax.plot( x , rvY0.cdf(x) , color = "blue" , label = r"$Y^0$" )
			for rvZ1 in rvZ1s:
				ax.plot( x , rvZ1.cdf(x) , color = "darkgreen" , label = r"$Z^1$" )
			ax.spines[['right', 'top']].set_visible(False)
			ax.legend( loc = "lower right" )
			ax.set_xlabel(r"$x$")
			ax.set_yticks([])
			
			## Figsize
			mm = 1. / 25.4
			pt = 1. / 72
			
			width  = 180*mm
			w_l    = 30*pt
			w_m    =  1*pt
			w_r    =  1*pt
			w_ax   = (width - (w_l + w_m + w_r)) / 2
			widths = [w_l,w_ax,w_m,w_ax,w_r]
			
			h_ax    = w_ax / (4/3)
			h_t     =  1*pt
			h_m     =  1*pt
			h_b     = 30*pt
			heights = [h_t,h_ax,h_m,h_ax,h_b]
			height  = sum(heights)
			
			grid.set_height_ratios(heights)
			grid.set_width_ratios(widths)
			fig.set_figheight(height)
			fig.set_figwidth(width)
			plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
			plt.savefig( os.path.join( self.opath , f"{self.prefix}_alphapooling.png" ) , dpi = 600 )
			plt.close(fig)
		
	##}}}
	
	def test_alphaPooling_exp(self):##{{{
		
		## Parameters
		np.random.seed(42)
		size    = 1_000
		nmod    = 3
		locs0   = np.linspace( -5 , 5 , nmod )
		locs1   = np.linspace( 5 , 15 , nmod )
		scales1 = np.linspace( 0.1 , 3 , nmod )
		scales0 = np.linspace( 0.5 , 5 , nmod )
		
		## Data
		Y0  =  np.random.normal( loc = 0 , scale = 1 , size = size )
		X0s = [np.random.exponential( scale = scales0[i] , size = size ) + locs0[i] for i in range(nmod)]
		X1s = [np.random.exponential( scale = scales1[i] , size = size ) + locs1[i] for i in range(nmod)]
		
		## mm correction
		mm  = bcmm.AlphaPooling( alpha = 1 ).fit( Y0 , X0s , X1s )
		Z1s,Z0s  = mm.predict( X1s , X0s )
		
		## Random variables
		rvY0  = bcs.rv_empirical.fit( Y0 )
		rvX0s = [ bcs.rv_empirical.fit( X0 ) for X0 in X0s ]
		rvX1s = [ bcs.rv_empirical.fit( X1 ) for X1 in X1s ]
		rvZ0s = [ bcs.rv_empirical.fit( Z0 ) for Z0 in Z0s ]
		rvZ1s = [ bcs.rv_empirical.fit( Z1 ) for Z1 in Z1s ]
		xmin  = min( [rv.a for rv in [rvY0] + rvX0s + rvX1s + rvZ0s + rvZ1s] )
		xmax  = max( [rv.b for rv in [rvY0] + rvX0s + rvX1s + rvZ0s + rvZ1s] )
		dx    = 0.05 * (xmax - xmin)
		x     = np.linspace( xmin - dx , xmax + dx , 1000 )
		
		## Plot
		if self.PLOTFIG:
			fig  = plt.figure()
			grid = mplg.GridSpec( 2 * 2 + 1 , 2 * 2 + 1 )
			
			ax  = fig.add_subplot(grid[1,1])
			ax.plot( x , rvY0.cdf(x) , color = "blue" , label = r"$Y^0$" )
			for rvX0 in rvX0s:
				ax.plot( x , rvX0.cdf(x) , color = "red" , label = r"$X^0$" )
			ax.spines[['right', 'top']].set_visible(False)
			ax.legend( loc = "lower right" )
			ax.set_xticks([])
			ax.set_ylabel("CDF")
			
			ax  = fig.add_subplot(grid[3,1])
			ax.plot( x , rvY0.cdf(x) , color = "blue" , label = r"$Y^0$" )
			for rvZ0 in rvZ0s:
				ax.plot( x , rvZ0.cdf(x) , color = "green" , label = r"$Z^0$" )
			ax.spines[['right', 'top']].set_visible(False)
			ax.legend( loc = "lower right" )
			ax.set_xlabel(r"$x$")
			ax.set_ylabel("CDF")
			
			ax  = fig.add_subplot(grid[1,3])
			ax.plot( x , rvY0.cdf(x) , color = "blue" , label = r"$Y^0$" )
			for rvX1 in rvX1s:
				ax.plot( x , rvX1.cdf(x) , color = "purple" , label = r"$X^1$" )
			ax.spines[['right', 'top']].set_visible(False)
			ax.legend( loc = "lower right" )
			ax.set_xticks([])
			ax.set_yticks([])
			
			ax  = fig.add_subplot(grid[3,3])
			ax.plot( x , rvY0.cdf(x) , color = "blue" , label = r"$Y^0$" )
			for rvZ1 in rvZ1s:
				ax.plot( x , rvZ1.cdf(x) , color = "darkgreen" , label = r"$Z^1$" )
			ax.spines[['right', 'top']].set_visible(False)
			ax.legend( loc = "lower right" )
			ax.set_xlabel(r"$x$")
			ax.set_yticks([])
			
			## Figsize
			mm = 1. / 25.4
			pt = 1. / 72
			
			width  = 180*mm
			w_l    = 30*pt
			w_m    =  1*pt
			w_r    =  1*pt
			w_ax   = (width - (w_l + w_m + w_r)) / 2
			widths = [w_l,w_ax,w_m,w_ax,w_r]
			
			h_ax    = w_ax / (4/3)
			h_t     =  1*pt
			h_m     =  1*pt
			h_b     = 30*pt
			heights = [h_t,h_ax,h_m,h_ax,h_b]
			height  = sum(heights)
			
			grid.set_height_ratios(heights)
			grid.set_width_ratios(widths)
			fig.set_figheight(height)
			fig.set_figwidth(width)
			plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
			plt.savefig( os.path.join( self.opath , f"{self.prefix}_alphapooling_exp.png" ) , dpi = 600 )
			plt.close(fig)
		
	##}}}

##}}}


##########
## main ##
##########

if __name__ == "__main__":
	## Custom parser to pass arguments for figures
	parser = argparse.ArgumentParser()
	parser.add_argument('--figures', action = "store_true" )
	parser.add_argument('unittest_args', nargs='*')
	args = parser.parse_args()
	sys.argv[1:] = args.unittest_args
	SBCKTestParameters.PLOTFIG = args.figures and has_mpl
	
	## And run
	unittest.main()
