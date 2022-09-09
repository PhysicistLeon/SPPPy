# -*- coding: utf-8 -*-
"""
"""

from materials import IsotropicMaterial

if __name__ == '__main__':
    mat = IsotropicMaterial(k=2,n=2)
    mat.parameters = [1,1]
    mat.n = 2
    
    print(f"{mat.n=}")
    print(f"{mat.k=}")
    print(f"{mat.parameters=}")
    print(f"eps = {mat.get_permittivity(1)}")
    
    # def merit_function(x):
    #     return x[0]**2 - x[1]**2
    
    # scipy.optimize.minimize(fun, [1,2])