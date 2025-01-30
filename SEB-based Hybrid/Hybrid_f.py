# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:31:51 2024

@author: dell
"""

# -*- coding: utf-8 -*-
""" Hybrid_f: A libray with Python functions for calculations of micrometeorological parameters.

    Meteorological functions:
        - es_calc:    Calculate saturation vapour pressures [Pa]
        - ea_calc:    Calculate actual vapour pressures [Pa]
        - vpd_calc:   Calculate vapour pressure deficits [Pa]
        - rho_calc:   Calculate air density [kg m-3]
        - ra_SEB_Inv:  Calculate aerodynamic resistance:invert the surface energy balance equation for rs [s/m]


@author: Zhang Chen <12214067@zju.edu.cn>
version: 1.0
date:    May 2024
"""


#First load python micrometeorological functions
import numpy as np
import math

def es_calc(ta= np.array([])):
    '''
    Function to calculate saturated vapour pressure from temperature.

    Input:
        - ta: (array of) measured air temperature [Celsius]
        
    Output:
        - es: (array of) saturated vapour pressure [Pa]
    '''

    # Determine length of array
    n = np.size(ta)
    # Check if we have a single (array) value or an array
    if n < 2:
        es = 0.61078*(math.e**((17.27*ta)/(ta+237.3)))
    else:   # Dealing with an array     
        # Initiate the output array
        es = np.zeros(n)
        # Calculate saturated vapour pressures, distinguish between water/ice
        for i in range(0, n):              
            es[i] = 0.61078*(math.e**((17.27*ta[i])/(ta[i]+237.3)))
    # Convert from kPa to Pa
    es = es * 1000.0
    return es # in Pa

def ea_calc(ta= np.array([]),\
            rh= np.array([])):
    '''
    Function to calculate actual saturation vapour pressure.

    Input:
        - ta: array of measured air temperatures [Celsius]
        - rh: Relative humidity [%]

    Output:
        - ea: array of actual vapour pressure [Pa]
    '''

    # Determine length of array
    n = np.size(ta)
    if n < 2:   # Dealing with single value...    
        # Calculate saturation vapour pressures
        es = es_calc(ta)
        # Calculate actual vapour pressure
        eact = float(rh) / 100.0 * es
    else:   # Dealing with an array
        # Initiate the output arrays
        eact = np.zeros(n)
        # Calculate saturation vapour pressures
        es = es_calc(ta)
        for i in range(0, n):
            # Calculate actual vapour pressure
            eact[i] = float(rh[i]) / 100.0 * es[i]
    return eact # in Pa

def vpd_calc(ta= np.array([]),\
             rh= np.array([])):
    '''
    Function to calculate vapour pressure deficit.

    Input:
        - ta: measured air temperatures [Celsius]
        - rh: (array of) rRelative humidity [%]
        
    Output:
        - vpd: (array of) vapour pressure deficits [Pa]
    '''

    # Determine length of array
    n = np.size(ta)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        # Calculate saturation vapour pressures
        es = es_calc(ta)
        eact = ea_calc(ta, rh) 
        # Calculate vapour pressure deficit
        vpd = es - eact
    else:   # Dealing with an array
        # Initiate the output arrays
        vpd = np.zeros(n)
        # Calculate saturation vapor pressures
        es = es_calc(ta)
        eact = ea_calc(ta, rh)
        # Calculate vapour pressure deficit
        for i in range(0, n):
            vpd[i] = es[i] - eact[i]
    return vpd # in Pa

def rho_calc(ta= np.array([]),\
             p= np.array([])):
    '''
    Function to calculate the density of air, rho, from air
    temperatures, relative humidity and air pressure.
    
    Input:
        - ta: (array of) air temperature data [Celsius]
        - p: (array of) air pressure data [Pa]
        
    Output:
        - rho: (array of) air density data [kg m-3]
    '''
    #from Pa to kPa
    p = p / 1000
    # Determine length of array
    n = np.size(ta)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        rho = p/(1.01*(ta+273)*0.287)
    else:   # Dealing with an array        
        # Initiate the output arrays
        rho = np.zeros(n)
        # calculate rho
        for i in range(0, n):
            rho[i] = p[i]/(1.01*(ta[i]+273)*0.287)
    return rho # in kg/m3

def ra_SEB_Inv(ta = np.array([]),\
               p = np.array([]),\
               ts = np.array([]),\
               Rn = np.array([]),\
               G = np.array([]),\
               LE = np.array([])):
    '''
    Function to calculate the aerodynamic resistance
    
    Input:
        - ta: (array of) air temperatures [C]
        - p: (array of) air pressure data [hPa]
        - ts: (array of) surface temperatures [C]
        - Rn: (array of) net radiation [W/m^2]
        - G: (array of) soil heat flux [W/m^2]
        - LE：(array of) latent heat flux [W/m^2]

    Output:
        - ra: (array of) aerodynamic resistance [s/m]
    
    Examples:
        >>> ra = ra_SEB_Inv(ta, p, ts, Rn, G, LE)
    '''
    
    # Determine length of array
    n = np.size(ta)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...

        p = p*100. # [Pa]
        rho = rho_calc(ta,p) # [kg/m3]
        cp = 1013 # [J kg-1 K-1]

        ra = rho*cp*(ts-ta)/(Rn-G-LE)
        
    else:   # Dealing with an array
        # Initiate output arrays
        ra = np.zeros(n)

        p = p*100. # [Pa]
        # Calculate rho and cp
        rho = rho_calc(ta,p) # [kg/m3]
        cp = 1013 # [J kg-1 K-1]

        # Calculate aerodynamic resistance        
        for i in range(0,n):
            ra[i] = rho[i]*cp*(ts[i]-ta[i])/(Rn[i]-G[i]-LE[i])
    return ra # [s/m]
