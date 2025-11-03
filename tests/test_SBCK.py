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

import os
import sys
import itertools as itt
import unittest
import argparse

import numpy as np
import scipy.stats as sc

import SBCK as bc
import SBCK.stats as bcs
import SBCK.datasets as bcd

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


#################################
## Stationary BC methods Tests ##
#################################

class StationaryBCTest(SBCKTestParameters):##{{{
    
    def __init__( self , corr_class , **kwargs ):##{{{
        super().__init__()
        self.corr_class  = corr_class
        self.corr_kwargs = kwargs
        self.metric      = bcs.wasserstein
    ##}}}
    
    def test_bc_calibration(self):##{{{
        
        ## Draw data
        np.random.seed(42)
        Y0,X0,_ = bcd.like_tas_pr(500)
        
        ## Correction
        corr = self.corr_class(**self.corr_kwargs).fit( Y0 , X0 )
        Z0 = corr.predict(X0)
        
        ## Check the correction in calibration period
        dYX0 = self.metric( Y0 , X0 )
        dYZ0 = self.metric( Y0 , Z0 )
        self.assertLess( dYZ0 , dYX0 )
        
    ##}}}
    
    def fig_BCX0(self):##{{{
        
        if self.PLOTFIG:
            ## Data
            np.random.seed(42)
            Y0,X0,X1 = bcd.gaussian_L_2d(10000)
            
            ## Correction
            corr = self.corr_class(**self.corr_kwargs).fit( Y0 , X0 , X1 )
            Z1 = corr.predict(X1)
            
            ## xylim
            xylim = [-5,11]
            dxy   = 0.1
            bins  = np.linspace( xylim[0] , xylim[1] , 100 )
            
            ## Figure
            titles = [r"$\mathbf{Y}^0$",r"$\mathbf{X}^0$",r"$\mathbf{Z}^1$",r"$\mathbf{X}^1$"]
            colors = ["blue","red","green","purple"]
            fig = plt.figure()
            grid = mplg.GridSpec(5,5+2)
            for iK,K in enumerate([Y0,X0,Z1,X1]):
                j = iK  % 2
                i = iK // 2
                ax  = fig.add_subplot(grid[2*i+1,2*j+1])
                im  = ax.hist2d( K[:,0] , K[:,1] , bins , density = True , vmin = 0 , vmax = 0.2 , cmap = plt.cm.Blues )
                ax.plot( K[:,0] , K[:,1] , color = colors[iK] , linestyle = "" , marker = "." , markersize = 0.05 )
                ax.text( xylim[1] - 3 * dxy , xylim[1] - 3 * dxy , titles[iK] , ha = "right" , va = "top" , fontdict = { "size" : 12 } , bbox = { 'facecolor' : 'none' , 'edgecolor' : 'black' , "boxstyle" : 'round' } )
                
                if i == 1:
                    ax.set_xlabel(r"$x_0$")
                else:
                    ax.set_xticks([])
                if j == 0:
                    ax.set_ylabel(r"$x_1$")
                else:
                    ax.set_yticks([])
                ax.spines[['right', 'top']].set_visible(False)
                ax.set_xlim(xylim)
                ax.set_ylim(xylim)
            
            ## Title
            ax = fig.add_subplot(grid[0,1:4])
            ax.text( 0 , 0 , self.name , ha = "center" , va = "center" , fontdict = { "weight" : "bold" , "size" : 12 } )
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.set_axis_off()
            
            ## Colorbar
            cax  = fig.add_subplot( grid[1:-1,5].subgridspec( 3 , 1 , height_ratios = [0.1,0.8,0.1] )[1,0] )
            cbar = plt.colorbar( mappable = im[-1] , cax = cax , ticks = [0,0.05,0.1,0.15,0.2] , label = "Density" )
            
            ## Figsize
            mm = 1. / 25.4
            pt = 1. / 72
            
            width  = 180*mm
            w_l    = 30*pt
            w_m    =  5*pt
            w_cl   =  5*pt
            w_cm   =  5*pt
            w_cr   = 35*pt
            w_ax   = (width - (w_l + w_m + w_cl + w_cm + w_cr)) / 2
            widths = [w_l,w_ax,w_m,w_ax,w_cl,w_cm,w_cr]
            
            h_ax    = w_ax
            h_t     = 15*pt
            h_m     =  5*pt
            h_b     = 30*pt
            heights = [h_t,h_ax,h_m,h_ax,h_b]
            height  = sum(heights)
            
            grid.set_height_ratios(heights)
            grid.set_width_ratios(widths)
            fig.set_figheight(height)
            fig.set_figwidth(width)
            plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
            
            plt.savefig( os.path.join( self.opath , f"{self.prefix}_fig_BCX0.png" ) , dpi = 600 )
            plt.close(fig)
    ##}}}
    
##}}}

class Test_QM(StationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):##{{{
        StationaryBCTest.__init__( self , bc.QM )
        unittest.TestCase.__init__( self , *args , **kwargs )
    ##}}}
    
    def test_wrapperQM(self):##{{{
        
        
        ## Start by define two kinds of laws from scipy.stats, the Normal and Exponential distribution
        np.random.seed(42)
        norm  = sc.norm
        expon = sc.expon
        
        ## And define calibration and reference dataset such that the law of each columns are reversed
        size = 10000
        Y0   = np.stack( [expon.rvs( scale = 2 , size = size ),norm.rvs( loc = 0 , scale = 1 , size = size )] ).T
        X0   = np.stack( [norm.rvs( loc = 0 , scale = 1 , size = size ),expon.rvs( scale = 1 , size = size )] ).T
        
        ## Generally, the law of Y0 and X0 is unknow, so we use the empirical histogram distribution
        qm0  = bc.QM( rvY0 = bcs.rv_empirical , rvX0 = bcs.rv_empirical ).fit( Y0 , X0 )
        Z0_h = qm0.predict(X0)
        
        ## Actually, this is the default behavior
        qm1= bc.QM().fit( Y0 , X0 )
        self.assertAlmostEqual( np.abs(Z0_h - qm1.predict(X0)).max() , 0. )
        
        ## In some case we know the kind of law of Y0 (or X0)
        qm2   = bc.QM( rvY0 = [expon,norm] ).fit( Y0 , X0 )
        Z0_Y0 = qm2.predict(X0)
        
        ## Or, even better, we know the law of the 2nd component of Y0 (or X0)
        qm3   = bc.QM( rvY0 = [expon,norm(loc=0,scale=1)] ).fit( Y0 , X0 )
        Z0_Y2 = qm3.predict(X0)
        
        ## Obviously, we can mix all this strategy to build a custom Quantile Mapping
        qm4   = bc.QM( rvY0 = [bcs.rv_empirical,norm(loc=0,scale=1)] , rvX0 = [norm,bcs.rv_empirical] ).fit( Y0 , X0 )
        Z0_Yh = qm4.predict(X0)
        
        ##
        if self.PLOTFIG:
            colors  = ["blue","red","green"]
            markers = ["o","x","+"]
            titles  = ["First component","Second component"]
            labels  = [r"Par. $Y^0$",r"Freeze $Y^0$","Mix"]
            fig = plt.figure()
            grid = mplg.GridSpec(3,5)
            for i in range(2):
                ax  = fig.add_subplot(grid[1,2*i+1])
                for iZ,Z in enumerate([Z0_Y0,Z0_Y2,Z0_Yh]):
                    ax.plot( Z0_h[:,i] , Z[:,i] , color = colors[iZ] , linestyle = "" , marker = markers[iZ] , label = labels[iZ] )
                ax.set_xlabel( "Empirical" )
                ax.set_ylabel( "Custom Quantile Mapping" )
                ax.set_title( titles[i] )
                ax.legend( loc = "upper left" , ncols = 1 )
                xylim    = list(ax.get_xlim())
                xylim[0] = min(xylim[0],ax.get_ylim()[0])
                xylim[1] = max(xylim[1],ax.get_ylim()[1])
                ax.plot( xylim , xylim , color = "black" )
                ax.set_xlim(xylim)
                ax.set_ylim(xylim)
            
            ## Figsize
            mm = 1. / 25.4
            pt = 1. / 72
            
            width  = 180*mm
            w_l    = 35*pt
            w_m    = 35*pt
            w_r    =  1*pt
            w_ax   = (width - (w_l + w_m + w_r)) / 2
            widths = [w_l,w_ax,w_m,w_ax,w_r]
            
            h_ax    = w_ax
            h_t     = 20*pt
            h_b     = 30*pt
            heights = [h_t,h_ax,h_b]
            height  = sum(heights)
            
            grid.set_height_ratios(heights)
            grid.set_width_ratios(widths)
            fig.set_figheight(height)
            fig.set_figwidth(width)
            plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
            plt.savefig( os.path.join( self.opath , f"{self.prefix}_test_wrapperQM.png" ) , dpi = 600 )
            plt.close(fig)
    ##}}}
    
##}}}


#####################################
## Non stationary BC methods Tests ##
#####################################

class NonStationaryBCTest(SBCKTestParameters):##{{{
    
    def __init__( self , tex_name , corr_class , **kwargs ):##{{{
        super().__init__()
        self.tex_name    = tex_name
        self.corr_class  = corr_class
        self.corr_kwargs = kwargs
        self.metric      = bcs.wasserstein
    ##}}}
    
    def test_bc_calibration(self):##{{{
        
        ## Draw data
        np.random.seed(42)
        Y0,X0,X1 = bcd.like_tas_pr(500)
        
        ## Correction
        corr = self.corr_class(**self.corr_kwargs).fit( Y0 , X0 , X1 )
        Z1,Z0 = corr.predict(X1,X0)
        
        ## Check the correction in calibration period
        dYX0 = self.metric( Y0 , X0 )
        dYZ0 = self.metric( Y0 , Z0 )
        self.assertLess( dYZ0 , dYX0 )
    ##}}}
    
    def test_fig_BCX1( self , size = 10_000 ):##{{{
        
        if self.PLOTFIG:
            ## Data
            np.random.seed(42)
            Y0,X0,X1 = bcd.gaussian_L_2d(size)
            
            ## Correction
            corr = self.corr_class(**self.corr_kwargs).fit( Y0 , X0 , X1 )
            Z1 = corr.predict(X1)
            
            ## xylim
            xylim = [-5,11]
            dxy   = 0.1
            bins  = np.linspace( xylim[0] , xylim[1] , 100 )
            
            ## Figure
            titles = [r"$\mathbf{Y}^0$",r"$\mathbf{X}^0$",r"$\mathbf{Z}^1$",r"$\mathbf{X}^1$"]
            colors = ["blue","red","green","purple"]
            fig = plt.figure()
            grid = mplg.GridSpec(5,5+2)
            for iK,K in enumerate([Y0,X0,Z1,X1]):
                j = iK  % 2
                i = iK // 2
                ax  = fig.add_subplot(grid[2*i+1,2*j+1])
                im  = ax.hist2d( K[:,0] , K[:,1] , bins , density = True , vmin = 0 , vmax = 0.2 , cmap = plt.cm.Blues )
                ax.plot( K[:,0] , K[:,1] , color = colors[iK] , linestyle = "" , marker = "." , markersize = 0.05 )
                ax.text( xylim[1] - 3 * dxy , xylim[1] - 3 * dxy , titles[iK] , ha = "right" , va = "top" , fontdict = { "size" : 12 } , bbox = { 'facecolor' : 'none' , 'edgecolor' : 'black' , "boxstyle" : 'round' } )
                
                if i == 1:
                    ax.set_xlabel(r"$x_0$")
                else:
                    ax.set_xticks([])
                if j == 0:
                    ax.set_ylabel(r"$x_1$")
                else:
                    ax.set_yticks([])
                ax.spines[['right', 'top']].set_visible(False)
                ax.set_xlim(xylim)
                ax.set_ylim(xylim)
            
            ## Title
            ax = fig.add_subplot(grid[0,1:4])
            ax.text( 0 , 0 , self.tex_name , ha = "center" , va = "center" , fontdict = { "weight" : "bold" , "size" : 12 } )
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.set_axis_off()
            
            ## Colorbar
            cax  = fig.add_subplot( grid[1:-1,5].subgridspec( 3 , 1 , height_ratios = [0.1,0.8,0.1] )[1,0] )
            cbar = plt.colorbar( mappable = im[-1] , cax = cax , ticks = [0,0.05,0.1,0.15,0.2] , label = "Density" )
            
            ## Figsize
            mm = 1. / 25.4
            pt = 1. / 72
            
            width  = 180*mm
            w_l    = 30*pt
            w_m    =  5*pt
            w_cl   =  5*pt
            w_cm   =  5*pt
            w_cr   = 35*pt
            w_ax   = (width - (w_l + w_m + w_cl + w_cm + w_cr)) / 2
            widths = [w_l,w_ax,w_m,w_ax,w_cl,w_cm,w_cr]
            
            h_ax    = w_ax
            h_t     = 17*pt
            h_m     =  5*pt
            h_b     = 30*pt
            heights = [h_t,h_ax,h_m,h_ax,h_b]
            height  = sum(heights)
            
            grid.set_height_ratios(heights)
            grid.set_width_ratios(widths)
            fig.set_figheight(height)
            fig.set_figwidth(width)
            plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
            
            plt.savefig( os.path.join( self.opath , f"{self.prefix}_fig_BCX1.png" ) , dpi = 600 )
            plt.close(fig)
    ##}}}
    
##}}}

class Test_CDFt(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):##{{{
        NonStationaryBCTest.__init__( self , r"CDF-$t$" , bc.CDFt )
        unittest.TestCase.__init__( self , *args , **kwargs )
    ##}}}
    
    def test_norm_oob(self):##{{{
        
        ## Parameters
        norms       = ["None","origin","dynamical"]
        oobs        = ["None","CC1","CC5","Y0","Y0CC"]
        nnorms      = len(norms)
        noobs       = len(oobs)
        corr_kwargs = { "norm" : None , "oob" : None }
        
        ## Data
        np.random.seed(42)
        Y0,X0,X1 = bcd.gaussian_L_2d(50000)
        
        
        ## Loop on columns
        for col in range(2):
            
            ##
            bins = np.linspace( -5 , 11 , 500 )
            x    = np.linspace( -5 , 11 , 1000 )
            rvY0 = bcs.rv_density.fit( Y0[:,col] )
            rvX0 = bcs.rv_density.fit( X0[:,col] )
            rvX1 = bcs.rv_density.fit( X1[:,col] )
            
            ## Fig parameters
            colors = ["blue","red","purple","green"]
            
            ## Figure
            if self.PLOTFIG:
                fig = plt.figure( dpi = 120 )
                grid = mplg.GridSpec( 1 + 2 * noobs + 1 , 1 + 2 * nnorms + 1 )
            
            ## Loop on corrections
            for (inorm,norm),(ioob,oob) in itt.product(enumerate(norms),enumerate(oobs)):
                
                ## Correction
                corr_kwargs["norm"] = norm
                corr_kwargs["oob"]  = oob
                corr = self.corr_class(**corr_kwargs).fit( Y0 , X0 , X1 )
                Z1 = corr.predict(X1)
                rvZ1 = bcs.rv_density.fit( Z1[:,col] )
                
                if self.PLOTFIG:
                    ## Figure
                    ax = fig.add_subplot( grid[1+2*ioob+1,1+2*inorm+1] )
                    ax.hist( Z1[:,col] , bins , color = "green" , alpha = 0.5 , density = True )
                    for iK,(K,rvK) in enumerate(zip([Y0,X0,X1,Z1],[rvY0,rvX0,rvX1,rvZ1])):
                        Kn = K[:,col].min()
                        Kx = K[:,col].max()
                        x  = np.linspace( Kn, Kx , 1000 )
                        ax.plot( x , rvK.pdf(x) , color = colors[iK] )
                    
                    if ioob < noobs - 1:
                        ax.set_xticks([])
                    else:
                        ax.set_xlabel(r"$x$")
                    if inorm == 0:
                        ax.set_ylabel("Density")
                    else:
                        ax.set_yticks([])
                    ax.set_xlim(-6,12)
                    ax.set_ylim(0,0.65)
                    
                    ax.spines[['right', 'top']].set_visible(False)
            
            if self.PLOTFIG:
                for inorm,norm in enumerate(norms):
                    ax = fig.add_subplot( grid[0,1+2*inorm+1] )
                    ax.text( 0 , 0 , norm , ha = "center" , va = "center" , fontdict = { "family" : "monospace" , "weight" : "bold" , "size" : 9 } )
                    ax.set_xlim(-1,1)
                    ax.set_ylim(-1,1)
                    ax.set_axis_off()
                
                for ioob,oob in enumerate(oobs):
                    ax = fig.add_subplot( grid[1+2*ioob+1,0] )
                    ax.text( 0 , 0 , oob , rotation = 90 , ha = "center" , va = "center" , fontdict = { "family" : "monospace" , "weight" : "bold" , "size" : 9 } )
                    ax.set_xlim(-1,1)
                    ax.set_ylim(-1,1)
                    ax.set_axis_off()
            
            
            if self.PLOTFIG:
                ## Figsize
                mm = 1. / 25.4
                pt = 1. / 72
                
                width  = 210*mm
                w_ll   = 12*pt
                w_l    = 30*pt
                w_m    =  5*pt
                w_r    =  5*pt
                w_ax   = (width - (w_ll + w_l + w_m + w_r)) / nnorms
                widths = [w_ll,w_l]
                for _ in range(nnorms):
                    widths.append(w_ax)
                    widths.append(w_m)
                widths[-1] = w_r
                
                h_ax    = w_ax / (16 / 11)
                h_tit   = 12*pt
                h_t     =  1*pt
                h_m     =  5*pt
                h_b     = 30*pt
                heights = [h_tit,h_t]
                for _ in range(noobs):
                    heights.append(h_ax)
                    heights.append(h_m)
                heights[-1] = h_b
                height  = sum(heights)
                
                grid.set_height_ratios(heights)
                grid.set_width_ratios(widths)
                fig.set_figheight(height)
                fig.set_figwidth(width)
                
                plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
                plt.savefig( os.path.join( self.opath , f"{self.prefix}_col{col}_norm_oob.png" ) , dpi = 600 )
                plt.close(fig)
        
    ##}}}
    
##}}}

class Test_R2D2(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):
        NonStationaryBCTest.__init__( self , r"R$^2$D$^2$" , bc.R2D2 )
        unittest.TestCase.__init__( self , *args , **kwargs )
    
##}}}

class Test_R2D2Reverse(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):
        NonStationaryBCTest.__init__( self , r"R$^2$D$^2$" , bc.R2D2 , start_by_margins = False )
        unittest.TestCase.__init__( self , *args , **kwargs )
    
##}}}

class Test_dOTC(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):
        NonStationaryBCTest.__init__( self , "dOTC" , bc.dOTC , cov_factor = 'std' )
        unittest.TestCase.__init__( self , *args , **kwargs )
    
    def test_fig_BCX1(self):
        NonStationaryBCTest.test_fig_BCX1( self , 1_000 )
    
##}}}

class Test_dOTC1d(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):
        NonStationaryBCTest.__init__( self , "dOTC1d" , bc.dOTC1d )
        unittest.TestCase.__init__( self , *args , **kwargs )
    
##}}}

class Test_dTSMBC(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):
        NonStationaryBCTest.__init__( self , "TSMBC" , bc.dTSMBC , lag = 3 , cov_factor = 'std' )
        unittest.TestCase.__init__( self , *args , **kwargs )
    
    def test_fig_BCX1(self):
        NonStationaryBCTest.test_fig_BCX1( self , 1_000 )
##}}}

class Test_IdBC(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):
        NonStationaryBCTest.__init__( self , "IdBC" , bc.IdBC )
        unittest.TestCase.__init__( self , *args , **kwargs )
    
    def test_bc_calibration(self):
        pass
    
##}}}

class Test_RBC(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):
        NonStationaryBCTest.__init__( self , "RBC" , bc.RBC )
        unittest.TestCase.__init__( self , *args , **kwargs )
    
##}}}

class Test_QQD(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):
        NonStationaryBCTest.__init__( self , r"QQD" , bc.QQD )
        unittest.TestCase.__init__( self , *args , **kwargs )
    
##}}}

class Test_QDM(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):
        NonStationaryBCTest.__init__( self , r"QDM" , bc.QDM )
        unittest.TestCase.__init__( self , *args , **kwargs )
    
##}}}

class Test_MBCn(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):
        NonStationaryBCTest.__init__( self , r"MBCn" , bc.MBCn )
        unittest.TestCase.__init__( self , *args , **kwargs )
    
    def test_fig_BCX1(self):
        NonStationaryBCTest.test_fig_BCX1( self , 1_000 )
    
##}}}

class Test_MRec(NonStationaryBCTest,unittest.TestCase):##{{{
    
    def __init__( self , *args, **kwargs ):
        NonStationaryBCTest.__init__( self , r"MRec" , bc.MRec )
        unittest.TestCase.__init__( self , *args , **kwargs )
    
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
