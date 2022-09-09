# -*- coding: utf-8 -*-
"""

"""
from typing import Annotated

class IsotropicMaterial:
    """ Изтропный материал
    """
    def __init__(self, n, k):
        self.n = n
        self.k = k

    def __getattr__(self, name):
        return self.__dict__[name]
        
    def __setattr__(self, name, value):
        if name == "parameters":
            self.__dict__['n'] = value[0]
            self.__dict__['k'] = value[1]
        else:
            self.__dict__[name] = value

    def get_permittivity(self, wavelength: Annotated[float, "meter"]) -> complex:
        n = self.n
        k = self.k
        return  n*n - k*k + 1j*2*n*k
    
class DrudeMaterial(IsotropicMaterial):
    def get_permittivity(self, wavelength: Annotated[float, "meter"]):
        return 0