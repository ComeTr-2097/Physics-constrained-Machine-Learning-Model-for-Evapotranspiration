# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:31:51 2024

@author: dell
"""

# -*- coding: utf-8 -*-
""" Hybrid_f: A libray with Python functions for calculations of micrometeorological parameters.

    Miscellaneous functions:
        - date2doy:   Calculates day of year from day, month and year data

    Meteorological functions: 
        - es_calc:    Calculate saturation vapour pressures [Pa]
        - ea_calc:    Calculate actual vapour pressures [Pa]
        - vpd_calc:   Calculate vapour pressure deficits [Pa]
        - Delta_calc: Calculate slope of vapour pressure curve [Pa K-1]
        - L_calc:     Calculate latent heat of vapourisation [J kg-1]
        - cp_calc:    Calculate specific heat [J kg-1 K-1]
        - gamma_calc: Calculate psychrometric constant [Pa K-1]
        - rho_calc:   Calculate air density [kg m-3]
        - ra_calc:    Calculate aerodynamic resistance from 
                      windspeed and roughness parameters (with/without vegetation cover) [s/m]
        - rs_calc:    Calculate surface resistance?
        - rs_PM_Inv:  Calculate surface resistance:invert the PM equation for rs [s/m]


@author: Zhang Chen <12214067@zju.edu.cn>
version: 2.0
date:    March 2024
"""




#First load python micrometeorological functions
import numpy as np
import math


def es_calc(airtemp= np.array([])):
    '''
    Function to calculate saturated vapour pressure from temperature.

    Input:
        - airtemp: (array of) measured air temperature [Celsius]
        
    Output:
        - es: (array of) saturated vapour pressure [Pa]
    '''

    # Determine length of array
    n = np.size(airtemp)
    # Check if we have a single (array) value or an array
    if n < 2:
        es = 0.61078*(math.e**((17.27*airtemp)/(airtemp+237.3)))
    else:   # Dealing with an array     
        # Initiate the output array
        es = np.zeros(n)
        # Calculate saturated vapour pressures, distinguish between water/ice
        for i in range(0, n):              
            es[i] = 0.61078*(math.e**((17.27*airtemp[i])/(airtemp[i]+237.3)))
    # Convert from kPa to Pa
    es = es * 1000.0
    return es # in Pa

def ea_calc(airtemp= np.array([]),\
            rh= np.array([])):
    '''
    Function to calculate actual saturation vapour pressure.

    Input:
        - airtemp: array of measured air temperatures [Celsius]
        - rh: Relative humidity [%]

    Output:
        - ea: array of actual vapour pressure [Pa]
    '''

    # Determine length of array
    n = np.size(airtemp)
    if n < 2:   # Dealing with single value...    
        # Calculate saturation vapour pressures
        es = es_calc(airtemp)
        # Calculate actual vapour pressure
        eact = float(rh) / 100.0 * es
    else:   # Dealing with an array
        # Initiate the output arrays
        eact = np.zeros(n)
        # Calculate saturation vapour pressures
        es = es_calc(airtemp)
        for i in range(0, n):
            # Calculate actual vapour pressure
            eact[i] = float(rh[i]) / 100.0 * es[i]
    return eact # in Pa

def vpd_calc(airtemp= np.array([]),\
             rh= np.array([])):
    '''
    Function to calculate vapour pressure deficit.

    Input:
        - airtemp: measured air temperatures [Celsius]
        - rh: (array of) rRelative humidity [%]
        
    Output:
        - vpd: (array of) vapour pressure deficits [Pa]
    '''

    # Determine length of array
    n = np.size(airtemp)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        # Calculate saturation vapour pressures
        es = es_calc(airtemp)
        eact = ea_calc(airtemp, rh) 
        # Calculate vapour pressure deficit
        vpd = es - eact
    else:   # Dealing with an array
        # Initiate the output arrays
        vpd = np.zeros(n)
        # Calculate saturation vapor pressures
        es = es_calc(airtemp)
        eact = ea_calc(airtemp, rh)
        # Calculate vapour pressure deficit
        for i in range(0, n):
            vpd[i] = es[i] - eact[i]
    return vpd # in Pa

def Delta_calc(airtemp= np.array([])):
    '''
    Function to calculate the slope of the temperature - vapour pressure curve

    Input:
        - airtemp: (array of) air temperature [Celsius]
    
    Output:
        - Delta: (array of) slope of saturated vapour curve [Pa K-1]
    '''

    # Determine length of array
    n = np.size(airtemp)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        # calculate vapour pressure
        es = es_calc(airtemp) # in Pa
        # Convert es (Pa) to kPa
        es = es / 1000.0
        # Calculate Delta
        Delta = es * 4098.0 / math.pow((airtemp + 237.3), 2)*1000
    else:   # Dealing with an array         
        # Initiate the output arrays
        Delta = np.zeros(n)
        # calculate vapour pressure
        es = es_calc(airtemp) # in Pa
        # Convert es (Pa) to kPa
        es = es / 1000.0
        # Calculate Delta
        for i in range(0, n):
            Delta[i] = es[i] * 4098.0 / math.pow((airtemp[i] + 237.3), 2)*1000
    return Delta # in Pa/K

def gamma_calc(airpress=np.array([])):
    '''
    Function to calculate the psychrometric constant gamma.

    Input:
        - airpress: array of air pressure data [Pa]
        
    Output:
        - gamma: array of psychrometric constant values [Pa K-1]
    '''

    #from pa to kpa
    airpress = airpress / 1000
    # Determine length of array
    n = np.size(airpress)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        # Calculate gamma
        gamma = 0.665 * airpress
    else:   # Dealing with an array
        # Initiate the output arrays
        gamma = np.zeros(n)
        for i in range(0, n):
            gamma[i] = 0.665 * airpress[i]
    return gamma # in Pa/K

def rho_calc(airtemp= np.array([]),\
             airpress= np.array([])):
    '''
    Function to calculate the density of air, rho, from air
    temperatures, relative humidity and air pressure.
    
    Input:
        - airtemp: (array of) air temperature data [Celsius]
        - airpress: (array of) air pressure data [Pa]
        
    Output:
        - rho: (array of) air density data [kg m-3]
    '''
    #from Pa to kPa
    airpress = airpress / 1000
    # Determine length of array
    n = np.size(airtemp)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        rho = airpress/(1.01*(airtemp+273)*0.287)
    else:   # Dealing with an array        
        # Initiate the output arrays
        rho = np.zeros(n)
        # calculate rho
        for i in range(0, n):
            rho[i] = airpress[i]/(1.01*(airtemp[i]+273)*0.287)
    return rho # in kg/m3


def ra_calc(h = np.array([]),\
            zm = np.array([]),\
            zh = np.array([]),\
            u = np.array([])):
    '''
    Function to calculate the aerodynamic resistance 
    (in s/m) from windspeed and height/roughness values
    Source: FAO Penman-Monteith equation, https://www.fao.org/3/X0490E/x0490e06.htm#TopOfPage
            The equation is restricted for neutral stability conditions,
            i.e., where temperature, atmospheric pressure, and wind velocity distributions follow nearly adiabatic conditions (no heat exchange).
            
            The application of the equation for short time periods (hourly or less) may require the inclusion of corrections for stability
    
    Input (measured at 2 m height):
        - h :  vegetation canopy height [m]
        - zm:  height of wind measurements [m]
        - zh:  height of humidity measurements [m]
        - u:   wind speed at height zm [m s-1]
    
    Fomula:
        - ra = [(ln((zm-d)/zom))*(ln((zh-d)/zoh))]/(k^2*u)
        
        1-Factors:[with vegetation cover]
        - d :  zero plane displacement height [m], d = 2/3*h
        - zom: roughness length governing momentum transfer [m], zom = 0.1*h
        - zoh: roughness length governing transfer of heat and vapour [m], zoh = 0.1*zom = 0.01*h
        (Source: Lin, C. J., P. Gentine, Y. F. Huang, K. Y. Guan, H. Kimm and S. Zhou (2018).
        "Diel ecosystem conductance response to vapor pressure deficit is suboptimal and independent of soil moisture.
        "Agricultural and Forest Meteorology 250: 24-34.)
        
        (Source: "Evaluation of optical remote sensing to estimate actual evapotranspiration and canopy conductance"
         "Remote Sensing of Environment")
        - k:   von Karman's constant, 0.41 [-]
        
        2-Factors:[without vegetation cover]
        - d :  zero plane displacement height [m], d = 0
        - zom: roughness length governing momentum transfer [m], zom = 0.01
        - zoh: roughness length governing transfer of heat and vapour [m], ln(zom/zoh) = 2; zoh = e^(ln(zom)-2)
        (Source: S. Liu, D. Mao, L. Lu. Measurement and estimation of the aerodynamic resistance.
                Hydrology and Earth System Sciences Discussions, 2006, 3 (3), pp.681-705. ffhal-00298684)
        - k:   von Karman's constant, 0.41 [-]

    Output:
        - ra: (array of) aerodynamic resistances [s/m]
    '''

    n = np.size(u)
    
    if n < 2: # Dealing with single value
        if h > 0: # With vegetation cover
            d = (2/3)*h
            zom = 0.1*h
            zoh = 0.01*h
            ra = ((np.log((zm-d)/zom))*(np.log((zh-d)/zoh)))/(0.1681*u) #0.41^2=0.1681
        else: # Without vegetation cover
            d = 0
            zom = 0.01
            zoh = zom * np.exp(-2)
            ra = ((np.log((zm-d)/zom))*(np.log((zh-d)/zoh)))/(0.1681*u) #0.41^2=0.1681
    
    else: # Dealing with an array
        d = np.zeros(n)
        zom = np.zeros(n)
        zoh = np.zeros(n)
        ra = np.zeros(n)
        for i in range(0,n):
            if h[i] > 0: # With vegetation cover
                d[i] = (2/3)*h[i]
                zom[i] = 0.1*h[i]
                zoh[i] = 0.01*h[i]
                ra[i] = ((np.log((zm[i]-d[i])/zom[i]))*(np.log((zh[i]-d[i])/zoh[i])))/(0.1681*u[i]) #0.41^2=0.1681
            else: # Without vegetation cover
                d[i] = 0
                zom[i] = 0.01
                zoh[i] = zom[i] * np.exp(-2)
                ra[i] = ((np.log((zm[i]-d[i])/zom[i]))*(np.log((zh[i]-d[i])/zoh[i])))/(0.1681*u[i]) #0.41^2=0.1681
    return ra # aerodynamic resistanc in s/m


def rs_PM_Inv(airtemp = np.array([]),\
        rh = np.array([]),\
        airpress = np.array([]),\
        Rn = np.array([]),\
        G = np.array([]),\
        LE = np.array([]),\
        h = np.array([]),\
        zm = np.array([]),\
        zh = np.array([]),\
        u = np.array([])):
    '''
    Function to calculate the surface
    (in mm) Monteith, J.L. (1965) Evaporation and environment.
    Symp. Soc. Exp. Biol. 19, 205-224
    
    Input:
        - airtemp: (array of) daily average air temperatures [C]
        - rh: (array of) daily average relative humidity values[%]
        - airpress: (array of) daily average air pressure data [hPa]
        - Rn: (array of) average daily net radiation [W/m^2]
        - G: (array of) average daily soil heat flux [W/m^2]
        - LE：(array of) average daily latent heat flux [W/m^2]
        - h :  vegetation canopy height [m]
        - zm:  height of wind measurements [m]
        - zh:  height of humidity measurements [m]
        - u:   wind speed at height zm [m s-1]

    Output:
        - rs: (array of) surface resistance [s/m]
    
    Examples:
        >>> rs = rs_calc(T,RH,press,Rn,G,LE,h,zm,zh,u)
    '''
    
    # Determine length of array
    n = np.size(airtemp)
    # Check if we have a single value or an array
    if n < 2:   # Dealing with single value...
        
        airpress = airpress*100. # [Pa]
        
        # Calculate Delta, gamma, lambda, rho and cp
        DELTA = Delta_calc(airtemp)/100. # [hPa/K]
        gamma = gamma_calc(airpress)/100. # [hPa/K]
        rho = rho_calc(airtemp,airpress) # [kg/m3]
        # cp = 1013 # [J kg-1 K-1]
        
        # Calculate saturated and actual water vapour pressures
        es = es_calc(airtemp)/100. # [hPa]
        ea = ea_calc(airtemp,rh)/100. # [hPa]
        
        # Calculate aerodynamic resistance
        ra = ra_calc(h,zm,zh,u) #[s/m]
        
        rs = ((((DELTA*(Rn-G)+rho*1013*(es-ea)/ra)/LE)-DELTA)/gamma-1.)*ra
    else:   # Dealing with an array  
        # Initiate output arrays
        rs = np.zeros(n)
                
        airpress = airpress*100. # [Pa]
        
        # Calculate Delta, gamma, lambda, rho and cp
        DELTA = Delta_calc(airtemp)/100. # [hPa/K]
        gamma = gamma_calc(airpress)/100. # [hPa/K]
        rho = rho_calc(airtemp,airpress) # [kg/m3]
        # cp = 1013 # [J kg-1 K-1]
        
        # Calculate saturated and actual water vapour pressures
        es = es_calc(airtemp)/100. # [hPa]
        ea = ea_calc(airtemp,rh)/100. # [hPa]
        
        # Calculate aerodynamic resistance
        ra = ra_calc(h,zm,zh,u) #[s/m]
        
        for i in range(0,n):
            rs[i] = ((((DELTA[i]*(Rn[i]-G[i])+rho[i]*1013*(es[i]-ea[i])/ra[i])/LE[i])-DELTA[i])/gamma[i]-1.)*ra[i]
    return rs # [s/m]
