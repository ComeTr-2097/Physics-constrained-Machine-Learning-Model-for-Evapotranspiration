# -*- coding: utf-8 -*-
"""
Hybrid_main: A libray with Python functions for calculations of actual evapotranspiration.

        Functions: 
        - ETa:    Calculate actual evapotranspiration

@author: Chen Zhang(12214067@zju.edu.cn)
version: 1.0
date:    May 2024
"""

#First load python micrometeorological functions
import Hybrid_f
import math

import pandas as pd
import numpy as np


#Calculate surface energy balance evaporation (actual)
def ETa(ta = np.array([]),\
        p = np.array([]),\
        ts = np.array([]),\
        Rn = np.array([]),\
        G = np.array([]),\
        ra = np.array([])):
    '''
    
    Input (measured at 2 m height):
        - ta: (array of) air temperatures [C]
        - p: (array of) air pressure [hpa]
        - ts: (array of) surface temperatures [C]
        - Rn: (array of) net radiation [W m-2]
        - G: (array of) soil heat flux [W m-2]
        - ra: aerodynamic resistance [s/m]

    Output:
        - Ep_seb: (array of) surface energy balance evaporation values [mm]
    
    Examples:
        >>> Ep_seb_data = Ep(ta,p,ts,Rn,G,ra)    
    '''

    
    # Determine length of array
    l = np.size(ta)
    # Check if we have a single value or an array
    if l < 2:   # Dealing with single value...

        # Calculate Delta, gamma and lambda
        p=p*100. # [Pa]

        # Lambda = 2.45e6 # [J/kg]
        cp = 1013
        rho = Hybrid_f2_SEB.rho_calc(ta,p)

        le = Rn-G-(rho*cp*(ts-ta))/ra
        # Epm = (Rn-G-(rho*cp*(ts-ta))/ra)/Lambda*86400 #kg m-2 s-1 to mm / day
    else:   # Dealing with an array  
        # Initiate output arrays
        le = np.zeros(l)
        # Epm = np.zeros(l)
        p=p*100. # [Pa]

        # Lambda = 2.45e6 # [J/kg]
        cp = 1013
        rho = Hybrid_f2_SEB.rho_calc(ta,p)

        for i in range(0,l):
            le[i] = Rn[i]-G[i]-(rho[i]*cp*(ts[i]-ta[i]))/ra[i]
            # Epm[i] = (Rn[i]-G[i]-(rho[i]*cp*(ts[i]-ta[i]))/ra[i])/Lambda*86400 #kg m-2 s-1 to mm / day
    return le # actual ET in mm
