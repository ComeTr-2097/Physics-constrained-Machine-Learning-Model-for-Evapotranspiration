# -*- coding: utf-8 -*-
"""
Hybrid_main: A libray with Python functions for calculations of actual evapotranspiration.
        Functions: 
        - ETa:    Calculate actual evapotranspiration
@author: Zhang Chen <12214067@zju.edu.cn>
version: 1.0
date:    July 2023
"""

#First load python micrometeorological functions
import Hybrid_f2
import math

import pandas as pd
import numpy as np

#Calculate Penman-Monteith evaporation (actual)
def ETa(airtemp = np.array([]),\
        rh = np.array([]),\
        airpress = np.array([]),\
        Rn = np.array([]),\
        G = np.array([]),\
        h = np.array([]),\
        zm = np.array([]),\
        zh = np.array([]),\
        u = np.array([]),\
        rs = np.array([])):
    '''
    Function to calculate the Penman Monteith evaporation
    (in mm) Monteith, J.L. (1965) Evaporation and environment.
    Symp. Soc. Exp. Biol. 19, 205-224
    
    Input (measured at 2 m height):
        - airtemp: (array of) daily average air temperatures [C]
        - rh: (array of) daily average relative humidity values[%]
        - airpress: (array of) daily average air pressure data [hPa]
        - Rn: (array of) average daily net radiation [W m-2]
        - G: (array of) average daily soil heat flux [W m-2]
        - h :  vegetation canopy height [m]
        - zm:  height of wind measurements [m]
        - zh:  height of humidity measurements [m]
        - u:   wind speed at height zm [m s-1]
        - rs: surface resistance [s/m]

    Output:
        - Epm: (array of) Penman Monteith evaporation values [mm]
    
    Examples:
        >>> Epm_data = Epm(T,RH,press,Rn,G,ra,rs)    
    '''

    
    # Determine length of array
    l = np.size(airtemp)
    # Check if we have a single value or an array
    if l < 2:   # Dealing with single value...
        
        # Calculate Delta, gamma and lambda
        airpress=airpress*100. # [Pa]
        
        DELTA = Hybrid_f2.Delta_calc(airtemp)/100. # [hPa/K]
        gamma = Hybrid_f2.gamma_calc(airpress)/100. # [hPa/K]
        Lambda = 2.45e6 # [J/kg]
        cp = 1013
        rho = Hybrid_f2.rho_calc(airtemp,airpress)
        # Calculate saturated and actual water vapour pressures
        es = Hybrid_f2.es_calc(airtemp)/100. # [hPa]
        ea = Hybrid_f2.ea_calc(airtemp,rh)/100. # [hPa]
        # Calculate aerodynamic resistance
        ra = Hybrid_f2.ra_calc(h,zm,zh,u)
        le = (DELTA*(Rn-G)+rho*cp*(es-ea)/ra)/(DELTA+gamma*(1.+rs/ra))
        Epm = ((DELTA*(Rn-G)+rho*cp*(es-ea)/ra)/(DELTA+gamma*(1.+rs/ra)))/Lambda*86400 #kg m-2 s-1 to mm / day
    else:   # Dealing with an array  
        # Initiate output arrays
        le = np.zeros(l)
        Epm = np.zeros(l)
        airpress=airpress*100. # [Pa]
        
        DELTA = Hybrid_f2.Delta_calc(airtemp)/100. # [hPa/K]
        gamma = Hybrid_f2.gamma_calc(airpress)/100. # [hPa/K]
        Lambda = 2.45e6 # [J/kg]
        cp = 1013
        rho = Hybrid_f2.rho_calc(airtemp,airpress)
        # Calculate saturated and actual water vapour pressures
        es = Hybrid_f2.es_calc(airtemp)/100. # [hPa]
        ea = Hybrid_f2.ea_calc(airtemp,rh)/100. # [hPa]
        # Calculate aerodynamic resistance
        ra = Hybrid_f2.ra_calc(h,zm,zh,u)
        
        for i in range(0,l):
            le[i] = (DELTA[i]*(Rn[i]-G[i])+rho[i]*cp*(es[i]-ea[i])/ra[i])/(DELTA[i]+gamma[i]*(1.+rs[i]/ra[i]))
            Epm[i] = ((DELTA[i]*(Rn[i]-G[i])+rho[i]*cp*(es[i]-ea[i])/ra[i])/(DELTA[i]+gamma[i]*(1.+rs[i]/ra[i])))/Lambda*86400 #kg m-2 s-1 to mm / day
    return le # actual ET in mm
