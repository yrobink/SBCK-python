
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

import warnings


############
## Typing ##
############

from typing import Any
from typing import Callable

###############
## Functions ##
###############

def deprecated( message: str ) -> Callable:##{{{
    """Decorator to raise a warning message with the warnings package.

    Parameters
    ----------
    message: str
        Message to pass to warnings

    Returns
    decorator: Callable
        The decorator

    """
    def decorator(func: Callable) -> Callable:
        def wrapper( *args: Any , **kwargs: Any ) -> Any:
            
            warnings.warn( message = message , category = DeprecationWarning )
            return func( *args , **kwargs )
        return wrapper
    return decorator
##}}}

