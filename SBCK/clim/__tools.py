
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

###############
## Libraries ##
###############

import itertools as itt
import numpy as np
import scipy.stats as sc


############
## Typing ##
############

from typing import Generator
from typing import Any


###############
## Functions ##
###############

## def yearly_window ##{{{

def yearly_window( ybeg_: int | str,
                   yend_: int | str,
                   wleft: int | str,
                   wpred: int | str,
                  wright: int | str,
                  tleft_: int | str,
                 tright_: int | str
                 ) -> Generator[tuple[str,str,str,str,str,str],Any,Any]:
    """Generator to iterate over years between ybeg_ and yend_, with a fitting
    window of lenght wleft + wpred + wright, and a centered predict window of
    length wpred.
    
    Arguments
    ---------
    ybeg_: int | str
        Starting year
    yend_: int | str
        Ending year
    wleft: int | str
        Lenght of left window
    wpred: int | str
        Lenght of middle / predict window
    wright: int | str
        Lenght of right window
    tleft_: int | str
        Left bound
    tright_: int | str
        Right bound
    
    Returns
    -------
    gen: Generator[tuple[str,str,str,str,str,str]]
        The generator
    
    Examples
    --------
    >>> ybeg_,yend_        = 2006,2100
    >>> wleft,wpred,wright = 5,10,5
    >>> tleft_,tright_     = 1951,2100
    >>> print( f"Iterate over {wleft}-{wpred}-{wright} window" )
    >>> print( " * L-bound / Fit-left / Predict-left / Predict-right / Fit-right / R-Bound" )
    >>> for tf0,tp0,tp1,tf1 in SBCK.tools.yearly_window( ybeg_ , yend_ , wleft, wpred , wright , tleft_ , tright_ ):
    >>>     print( f" *    {tleft_} /     {tf0} /         {tp0} /          {tp1} /      {tf1} /    {tright_}" )
    >>>
    >>> ## Output
    >>> ## Iterate over 5-10-5 window
    >>> ##  * L-bound / Fit-left / Predict-left / Predict-right / Fit-right / R-Bound
    >>> ##  *    1951 /     2001 /         2006 /          2015 /      2020 /    2100
    >>> ##  *    1951 /     2011 /         2016 /          2025 /      2030 /    2100
    >>> ##  *    1951 /     2021 /         2026 /          2035 /      2040 /    2100
    >>> ##  *    1951 /     2031 /         2036 /          2045 /      2050 /    2100
    >>> ##  *    1951 /     2041 /         2046 /          2055 /      2060 /    2100
    >>> ##  *    1951 /     2051 /         2056 /          2065 /      2070 /    2100
    >>> ##  *    1951 /     2061 /         2066 /          2075 /      2080 /    2100
    >>> ##  *    1951 /     2071 /         2076 /          2085 /      2090 /    2100
    >>> ##  *    1951 /     2081 /         2086 /          2095 /      2100 /    2100
    >>> ##  *    1951 /     2081 /         2096 /          2100 /      2100 /    2100
    """
    
    ybeg   = int(ybeg_)
    yend   = int(yend_)
    tleft  = int(tleft_)
    tright = int(tright_)
    
    tp0  = int(ybeg)
    tp1  = tp0 + wpred - 1
    tf0  = tp0 - wleft
    tf1  = tp1 + wright
    
    while not tp0 > yend:
        
        ## Work on a copy, original used for iteration
        rtf0,rtp0,rtp1,rtf1 = tf0,tp0,tp1,tf1
        
        ## Correction when the left window is lower than tleft
        if rtf0 < tleft:
            rtf1 = rtf1 + tleft - rtf0
            rtf0 = tleft
        
        ## Correction when the right window is upper than yend
        if rtf1 > tright:
            rtf1 = tright
            rtf0 = rtf0 - (tf1 - tright)
        if rtp1 > tright:
            rtp1 = tright
        
        ## The return
        yield [str(x) for x in [rtf0,rtp0,rtp1,rtf1]]
        
        ## And iteration
        tp0 = tp1 + 1
        tp1 = tp0 + wpred - 1
        tf0 = tp0 - wleft
        tf1 = tp1 + wright
##}}}


