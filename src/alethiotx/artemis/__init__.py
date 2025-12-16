"""
**Artemis**: public knowledge graphs enable accessible and scalable drug target discovery.

Github Repository   
-----------------

`GitHub - alethiotx/artemis-paper <https://github.com/alethiotx/artemis-paper>`_
"""

from .chembl import *
from .clinical import *
from .cv import *
from .hgnc import *
from .mesh import *
from .pathway import *
from .upset import *
from .utils import *

__all__ = ['chembl', 'clinical', 'cv', 'hgnc', 'mesh', 'pathway', 'upset', 'utils']