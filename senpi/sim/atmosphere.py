# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 09:59:44 2020

@author: cvalenta3
"""

import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy import constants as spc
import math
import pandas as pd
import sys
import datetime
from pysolar.solar import *
import warnings


def AngstromCalc(beta, alpha, lambdaL):
    # Computes aerosol optical depth using Angstrom's expression
    # Inputs
    #   beta - Angstrom turbidity coefficient at 1 um (beta>0.2 indicates heavy pollution, beta>0.4 indicates very heavy pollution)
    #   alpha - Angstrom Wavelength exponent (alpha<1 indicates course aerosols radius>0.5um - dust or sea spray; alpha>1 indicates fine mode aerosols radius<0.5um - urban pollution and biomass )
    #   lambdaL - wavelength to calculate AOD (meters)
    # Outputs
    #   AOD - aerosol optical depth at lambdaL

    AOD = beta * (lambdaL * 1e6) ** (-alpha)

    return AOD


def ElevationAdjustment(z, Profile, Angle):
    # Takes in a vertical profile and returns a profile along a slant path
    # Inputs
    #   z - Altitude/Range vector (meters)
    #   Profile -vertical profile of atmospheric constituent
    #   Angle - off zenith angle (degrees)
    # Outputs
    #   AltVec - altitude vector corresponding to range vector and Slant Profile (meters)
    #   SlantProfile - atmospheric profile at off-nadir angle

    AltVec = z * np.cos(Angle * np.pi / 180)
    indx = Sort(AltVec, z)

    if (Profile.ndim == 1):
        SlantProfile = np.array([])
        for indxx in indx:
            SlantProfile = np.append(SlantProfile, Profile[int(indxx)])
    elif(Profile.ndim == 2):
        SlantProfile = np.array([[]])
        if (np.shape(Profile)[0] == np.size(z)):
            SlantProfile = SlantProfile.reshape(0, np.shape(Profile)[1])
            for indxx in indx:
                SlantProfile = np.vstack([SlantProfile, Profile[int(indxx), :]])
        elif (np.shape(Profile)[1] == np.size(z)):
            SlantProfile = SlantProfile.reshape(0, np.shape(Profile)[0])
            for indxx in indx:
                SlantProfile = np.vstack([SlantProfile, Profile[:, int(indxx)]])
            SlantProfile = SlantProfile.T
        else:
            raise NameError('Incompatible dimensions')
    else:
        raise NameError('Cannot work for array dimensions>2')

    return [AltVec, SlantProfile]


def Sort(AltVec, z):
    # Helper function for Elevation Adjustment
    indx = np.array([])
    for a in AltVec:
        val=0.5
        count=0
        indxSearch=[]
        while (np.size(indxSearch)==0):
            indxSearch=np.argwhere(np.isclose(a,z,atol=val))
            val=2*val
            count=count+1
            if (count==10):
                raise NameError('Cannot find approximate value')
        indx = np.append(indx, indxSearch[0])  # possibility of more than 1 return. Always go with lowest

    return indx


def TotalPrecipitableWater(mixingRatioProfile, PressureProfile):
    # Calculates the total precipitable water in a column of air
    # Inputs
    #   mixingRatioProfile - vertical Water vapor mixing Ratio profile (kg/kg)
    #   PressureProfile - profile of pressures corresponding to mixing Ratio path (hPa)
    # Outputs
    # TPW - integrated water in vertical column (mm or kg/m^2)
    
    #Need to check this equation and process
    H20Density=1000 #kg/m^3
    TPW=-1/(H20Density*spc.g)*np.trapz(1000*mixingRatioProfile,100*PressureProfile) #multiply by 1000 to put Mixing ratio in g/kg to get TPW in mm and 100 by hPa to get Pa which is in SI units
    
    return TPW


def ScaleHeight(z, Tmean):
    # Calculates scale height profile
    # Inputs
    #   z - Altitude vector (meters)
    # TODO: Add full calculation that depends on temperature:
    #   H = kT/Mg
    #   where:
    #   R = 8.31446261815324 - Gas constant (J/K-mol)
    #   T - temperature (K)
    #   M = 28.97 - mean molecular mass of one mole of atmos particles
    #   g - acceleration due to gravity (m/s^2) can use USU code for this
    # Outputs
    #   H - atmospheric scale height profile (meters)

    # Molar Mass kg/mol
    N2_MolMass = 2 * 14.0067 / 1000
    O2_MolMass = 2 * 15.999 / 1000
    Ar_MolMass = 2 * 39.948 / 1000
    CO2_MolMass = (1 * 12.0107 + 2 * 15.999) / 1000
    Ne_MolMass = 2 * 20.1797 / 1000
    O3_MolMass = 3 * 15.999 / 1000
    He_MolMass = 2 * 4.002602 / 1000
    CH4_MolMass = (1 * 12.0107 + 4 * 1.00784) / 1000
    Kr_MolMass = 2 * 83.798 / 1000
    H2_MolMass = 2 * 1.00784 / 1000
    N20_MolMass = (2 * 14.0067 + 15.999) / 1000

    # http://acmg.seas.harvard.edu/people/faculty/djj/book/bookchap1.html
    # Table 1-1 Mixing ratios of gases in dry air
    # Gas, Mixing ratio (MR) (mol/mol)
    N2_MR = 0.78
    O2_MR = 0.21
    Ar_MR = 0.0093
    CO2_MR = 365E-6
    Ne_MR = 18E-6
    O3_MR = 1E-6  # varies 0.01E-6 to 10E-5
    He_MR = 5.2E-6
    CH4_MR = 1.7E-6
    Kr_MR = 1.1E-6
    H2_MR = 500E-9
    N2O_MR = 320E-9
    # What about water vapor?

    MR_list = np.array([N2_MR, O2_MR, Ar_MR, CO2_MR, Ne_MR, O3_MR, He_MR, CH4_MR, Kr_MR, H2_MR, N2O_MR])
    # Total_MR=np.sum(MR_list)  # value used for debugging

    atmos_MolMass = (N2_MR * N2_MolMass + O2_MR * O2_MolMass + Ar_MR * Ar_MolMass + CO2_MR * CO2_MolMass
                     + Ne_MR * Ne_MolMass + O3_MR * O3_MolMass + He_MR * He_MolMass + CH4_MR * CH4_MolMass
                     + Kr_MR * Kr_MolMass + H2_MR * H2_MolMass + N2O_MR * N20_MolMass
                     )

    # The value below is from some old code that John Stewart wrote. Not sure exactly where that value came from...
    # ATM molecular mass
    # atm_MolMass = .0289644;   # kg/Mol - sum of product of mixing ratio times molecular weight

    # calculate scale height
    # Where does this equation come from?

    H = (spc.Boltzmann * Tmean) / (atmos_MolMass * spc.g / spc.Avogadro)

    # Alternatively, could use the mean molecular mass, a constant below 100 km
    # BUT gravity is allowed to vary with location and altitude (comes from another def)
    # *** this is only true for DRY atmosphere, so might be better to use the above definition, if there's a way to add in water vapor
    # M = 0.028965
    # H = (spc.R * Tmean) / (M * gravity)

    return H


def numberDensity(z, H, n0):
    # Calculates a number density profile from a single sea level value
    # Inputs
    #   z - Altitude vector (meters)
    #   H - scale height profile (meters)
    #   n0 - number density of atmospheric molecules (N2+O2)
    #   AT SEA or GROUND LEVEL
    # Outputs
    #   n - number density profile (1/m^3)

    n = n0 * np.exp(-z / H)

    return n


def getNumberDensityProfile(PressureProfile, TempProfile):
    # Calculates a number density profile from a single sea level value
    # Inputs
    #   TempProfile - temperature profile of atmosphere (Kelvin)
    #   PressureProfile - output vector of pressure profile (hPascals)
    #   AT SEA or GROUND LEVEL
    # Outputs
    #   NDProfile - number density profile (1/m^3)

    # 100 converts hPascal to Pascal
    NDProfile = (100 * PressureProfile * spc.Avogadro) / (spc.R * TempProfile)

    return NDProfile


def makeTemperatureProfile(z, Tc, DALapse, H_tropo):
    # Generates a vertical profile of temperature
    # Inputs
    #   z - Altitude vector (meters)
    #   Tc - Ground level Temperature (Celcius)
    #   DALapse - (Degrees Kelvin/meter)
    #   H_tropo - height of the tropopause (meters)
    # Outputs
    #   TempProfile - temperature profile of atmosphere (Kelvin)

    # ToDo - add ability to not be at sea level
    groundLevelTempK = Tc + 273.15
    TempProfile = groundLevelTempK + DALapse * z
    val_index = (np.abs(z - H_tropo)).argmin()
    TempProfile[z > H_tropo] = TempProfile[val_index]

    return TempProfile


def getGasProfiles(NDProfile):
    # Generates atmospheric density profiles for Earth mixing ratios
    # Inputs
    #   NDProfile - number density profile (1/m^3)
    # Outputs
    #   GasProfile - number density profile of each atmospheric gas (1/m^3)

    # http://acmg.seas.harvard.edu/people/faculty/djj/book/bookchap1.html
    # Table 1-1 Mixing ratios of gases in dry air
    # Gas, Mixing ratio (MR) (mol/mol)
    N2_MR = 0.78
    O2_MR = 0.21
    Ar_MR = 0.0093
    CO2_MR = 365E-6
    Ne_MR = 18E-6
    O3_MR = 1E-6  # varies 0.01E-6 to 10E-5
    He_MR = 5.2E-6
    CH4_MR = 1.7E-6
    Kr_MR = 1.1E-6
    H2_MR = 500E-9
    N2O_MR = 320E-9
    # TODO What about water vapor?

    MR = np.array([N2_MR, O2_MR, Ar_MR, CO2_MR, Ne_MR, O3_MR, He_MR, CH4_MR, Kr_MR, H2_MR, N2O_MR])

    GasProfile = np.outer(MR, NDProfile)

    return GasProfile


def makePressureProfile(z, H, p_ground):
    # Generates a vertical profile of pressure
    # Inputs
    #   z - Altitude vector (meters)
    #   H - scale height (meters)
    #   p_ground - ground level value of pressure (hPascals)
    # Output
    #   PressureProfile - output vector of pressure profile (hPascals)

    prat = np.exp(-z / H)
    PressureProfile = prat * p_ground

    return PressureProfile


# =============================================================================
# The following two methods are two different ways of getting the molecular backscatter coefficient
# Rayleigh_RCS is only the backscatter cross section and will need to be multiplied by number density to get the backscatter coefficient.
#    This method allows you to account for wavelength and species
# MolecularBackscatter only allows you to account for wavelength
#
# These two methods and a third method that allows you to account for depol ratio are compared in:
# Beta_molecular_comparison.py
# =============================================================================

def Rayleigh_RCS(lambdaL, aSquared, gammaSquared):
    # Calculates Rayleigh Cross section
    # Equations from Chapter 9 of Lidar: Range-Resolved Optical Remote
    # Sensing of the Atmosphere by Claus Weitkemp
    # Inputs
    #   lambdaL - excitation wavelength (meters)
    #   aSquared - square of the mean polarizability of the molecular polarizability tensor (m^6)
    #   gammaSquared - square of the anisotropy of the molecular polarizability tensor (m^6)
    # Outputs
    #   dRCS - differential backscatter cross section (m^2/sr)

    adj = (4 * np.pi * spc.epsilon_0) ** 2
    k_nu = np.pi ** 2 / spc.epsilon_0 ** 2

    nuL = 1 / lambdaL

    # Equation 9.5: Cabannes Line
    dRCS_cab = k_nu * nuL ** 4 * (aSquared * adj + (7 / 180) * gammaSquared * adj)

    # Equation 9.6: pure rotational Raman
    dRCS_RR = k_nu * nuL ** 4 * ((7 / 60) * gammaSquared * adj)

    # Rayleigh diff cross section, sum of 9.5 and 9.6, (see p. 249 of Weitkamp)
    dRCS = dRCS_cab + dRCS_RR

    return dRCS


def MolecularBackscatter(n, lambdaL):
    # Calculates atmospheric Backscatter profile due to Rayleigh Scattering
    # Uses equation 2.134 from Measures, 1992
    # Inputs
    #   n - number density profile of N2+O2 molecules
    #   lambdaL - wavelength (meters)
    # Output
    #   bm - Rayleigh backscatter profile (1/meter-sr)

    # TODO: add atmospheric constituents
    # TODO: (potentially) add elevation

    betam = 5.45e-32 * (0.550E-6 / lambdaL) ** 4  # From Measures, (1984), p. 42
    bm = betam * n  # molecular volume backscatter coefficient in m^2/sr

    return bm


def RayleighBackscatter(N, lambdaL, delta):
    # This equation is given in many sources: Elterman, 1962; Stergis, 1966;
    # Goody and Yung, 1995; Kidder and VonderHaar, 1995; Measures Eqn 2.132
    # Inputs
    # n - number density profile of "air" molecules
    # lambdaL - wavelength (meters)
    # delta - depolarization factor
    # Output
    # RayBack - Rayleigh backscatter profile (1/meter-sr)
    # RaySca - Rayleigh total scattering coefficient (1/meter)
    #   -->Equilavent to molecular extinction for

    # Expression for refractive index of air from Elden, 1952/Liou, 1980: Eqn 3.3.17
    # This analytical expression assumes lambdaL between 0.2 and 20 um and a
    # homogeneous mixture of air molecules

    # TODO: Find source for species refractive indices and replace this
    # analytic equation with: m-1 (see Elterman, 1962)
    lambdaLm = lambdaL * 1e6
    term2 = 2949810 / (146 - lambdaLm ** (-2))
    term3 = 25540 / (41 - lambdaLm ** (-2))
    no_1 = (1e-8) * (6432.8 + term2 + term3)

    term1 = (32 * spc.pi ** 3) / (3 * lambdaL ** 4)
    term2 = (1 / N[0] ** 2)
    term3 = no_1 ** 2
    term4 = (6 + 3 * delta) / (6 - 7 * delta)

    sigma_sca = term1 * term2 * term3 * term4
    RaySca = N * sigma_sca

    # Calculate backscatter from total Rayleigh scatter using 8pi/3 factor
    sigma_back = (3 / (8 * spc.pi)) * sigma_sca
    RayBack = N * sigma_back

    return RayBack, RaySca

# =============================================================================
# End molecular backscatter methods
# =============================================================================


def MolecularExtinction(bm):
    # Calculates atmospheric extinction profile due to Rayleigh Scattering
    # Inputs
    #   bm - Rayleigh backscatter profile (1/meter-sr)
    # Output
    #   am - Rayleigh extinction profile (1/meter)
    # NOTE: The below conversion from molecular backscatter coefficient (bm) to
    #       molecular extinction coefficient assumes that there is no molecular
    #       absorption. Thus, the below equation is just converting molecular
    #       backscatter to total molecular scatter. This assumption is
    #       wavelength-dependent and will not hold for wavelengths that are
    #       strongly absorbed by atmospheric molecules/trace gases (e.g. O3, H20)
    
    #TODO: Set flags/trace errors for wavelengths that affected by molecular absorption
    #TODO: add atmospheric constituents
    
    am = (8 * np.pi / 3.0) * bm

    return am


def makeWaterVaporMixingRatioProfile(z, w, H20_ScaleHeight, D_height, D_depth, D_width, D_above):
    # Provides water vapor profile
    # pAssumes an exponential falloff above the PBL
    # Inputs
    #   z - Altitude vector (meters)
    #   w water vapor mixing ratio at ground level (kg/kg)
    #   H20_ScaleHeight - arbitrary scale height for H20 (meters)
    #   D_height - dry duct altitude start (meters)
    #   D_depth - dry duct depth (percent as a decimal reduction)
    #   D_width - dry duct width (meters)
    #   D_above - mixing ratio decrease above the dry duct  (percent as a decimal reduction)
    # Output
    #   WVmixingRatio - profile of Water Vapor Mixing Ratio (kg/kg)

    WVmixingRatio = w * np.exp(-z / H20_ScaleHeight)
    # Find values which fall in the duct range and replace with the percent reduction
    WVmixingRatio = np.where((z > D_height) & (z < (D_height + D_width)), WVmixingRatio * D_depth, WVmixingRatio)
    # Find values above duct and replace with the percent reduction
    WVmixingRatio = np.where(z >= (D_height + D_width), WVmixingRatio * D_above, WVmixingRatio)

    return WVmixingRatio


def AerosolExtinction(z, AOD, PBLHeight, AerosolScaleHeight):
    # Provides aerosol extinction profile given an AOD and PBL Height
    # Assumes an exponential falloff above the PBL
    # Inputs
    #   z - Altitude vector (meters)
    #   AOD - vertical aerosol optical depth
    #   PBLHeight - height of the planetary boundary layer (meters)
    #   AerosolScaleHeight - aerosol scale height above PBL (meters)
    # Output
    # aa - aerosol extinction profile (1/meter)
    
    
    #Sets the shape of the aerosol extinction profile to be constant in the PBL
    temp_aerosol_shape=np.zeros(np.size(z))
    temp_aerosol_shape[np.where(z<=PBLHeight)]=1 
    temp_aerosol_shape[np.where(z>PBLHeight)] = np.exp(-(z[np.where(z>PBLHeight)])/AerosolScaleHeight)
    
    #temp_aerosol_integral = np.trapz(temp_aerosol_shape, dx=res)
    
    temp_aerosol_integral = np.trapz(temp_aerosol_shape, z)#dx=res/2.)
    
    aa = temp_aerosol_shape*(AOD/temp_aerosol_integral)
    
    return aa


def AerosolBackscatter(Sa, aa):
    # Provides aerosol backscatter profile
    # Inputs
    #   Sa - extinction-to-backscatter/"lidar" ratio (depends on wavelength)
    #   aa - aerosol  extinction profile (1/meter)
    # Outputs
    # ba - aerosol backscatter profile (1/meter-sr)
 
    
    ba = aa/Sa

    return ba


def TotalBackscatter(bm, ba):
    # Calculates total backscatte profile
    # Inputs
    # bm - Rayleigh backscatter profile (1/meter-sr)
    # ba - aerosol backscatter profile (1/meter-sr)
    # Outputs
    # bt - total backscatter (1/meter-sr)

    bt = bm + ba

    return bt


def TotalExtinction(am, aa):
    # Calculates total backscatte profile
    # Inputs
    # aa - Rayleigh extinction profile (1/meter)
    # aa - aerosol extinction profile (1/meter)
    # Outputs
    # at - total backscatter (1/meter-sr)

    at = am + aa

    return at


def Transmittance(z, at):
    # Calculates one-way path transmittance profile
    # Inputs
    #   z - Altitude vector (meters)
    #   aa - aerosol extinction profile (1/meter)
    #   am - Rayleigh extinction profile (1/meter)
    # Outputs
    #   tm - transmittance profile (unitless < 1)
    res = z[1] - z[0]

    int_a = integrate.cumtrapz(at, dx=(res / 2), initial=at[0])

    # tm = np.exp(-2*int_a)  # two-way
    tm = np.exp(-int_a)  # one-way

    return tm


def getMixingRatioFromRH(RH, Tc, pa):
    # Code written by J. Stewart and converted to Python
    # Calculates water vapor mixing ratio given relative humidity, temperature, and pressure
    # inputs:
    #    RH is relative humidity (percent) - not a decimal!
    #    Tc is air temperature (degrees Kelvin)
    #    pa is air pressure mbar (hPascals) - note 1 hPa = 100 Pa
    # outputs:
    #    w water vapor mixing ratio (kg/kg)
    RH = RH / 100
    # saturation vapor pressure es in hPa
    # August-Roche-Magnus Formula
    # https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation#Meteorology_and_climatology
    es = 6.1094 * np.exp(17.625 * (Tc - 273.15) / ((Tc - 273.15) + 243.04))
    # saturation specific humidity
    # qs = epsillon * es ./ (pa + 0.622* es);
    # partial pressure of dry air
    # pa = rhoa * 286.9 * T  % (kg/m^3)(J/kg)
    # pw = rhow * 461.5 * T  %
    # Molecular Weight Air = 28.97 g/mol
    # Molecular Weight Water = 18.02 g/mol
    # Universal Gas Constant Ru 8.31447 J/mol
    # R = Ru / Mgas
    # epsillon = .622 is ratio of molecular Mwater / M air
    epsillon = 0.622
    # specific humidity same as mixing ratio
    # relative humidity is vapor pressure / saturation vapor pressure
    e = RH * es
    # note using equation above with e, Tc is the Dew point
    w = epsillon * e / (pa - e)  # http://glossary.ametsoc.org/wiki/Mixing_ratio

    return w


def SkySpectralRadiance(parent_dir, lambdaVec,
                        date, lat, lon,
                        surfaceTilt, surfaceDirection,
                        pressure, precipWater, ozone, ozonePeakHeight,
                        AODarray, g, r_g_lamb):
    """Calculates sky spectral radiance (W/m^2-sr-um)

    Based on the analytical model by Bird and Riordan, 1986 (BR1986)
    https://doi.org/10.1175/1520-0450(1986)025<0087:SSSMFD>2.0.CO;2

    Inputs
    ------
    parent_dir
       parent directory to find location of Standard.xlsx
    lambdaVec
       vector of wavelength in [m]
    date
      `datetime.datetime` object
    lat
      observer latitude in [°]
    lon
      observer longitude in [°]
    surfaceTilt
      angle of surface normal from vertical [°]
    surfaceDirection 
      bearing of surface normal from cardinal North [°]
    pressure
      local surface atmospheric pressure [mbar]
    precipWater
      integrated precipitable water [inches]
      (Visit SPC mesoanalysis for a value)
    ozone
      ozone concentration at location [Dobson Units - DU]
    ozonePeakHeight
      height of peak ozone concentration [km]
    AODarray
      set of Aerosol Optical Depth values from NASA Aeronet
      Station at wavelengths (in [nm]) of
      (340, 380, 440, 500, 675, 870, 1020, 1640)
    g
      aerosol asymmetry factor, g=0.65 for rural aerosol model (Bird & Riordan, 1986)
      For dry aerosols, g can vary from 0.54 to 0.67 (Zhao et al., 2018)
      For hydroscopic aerosols, g will vary more
    r_g_lamb
      ground albedo

    Outputs
    -------
    lambnew
      Wavelength range input [um]
    L
      direct + diffuse sky spectral radiance [W/m³/sr]
    L_s
      diffuse-only sky spectral radiance [W/m³/sr]
    Irrad
      Calculated irradiance on a horizontal surface [W/m^2/um]
    Irrad_tilt
      Calculated irradiance on a tilted surface [W/m^2/um]
    lamb
      Wavelength range from exo-atmospheric irradiance table [um]
    irradiances
      Irradiance given in exo-atmospheric irradiance table (TOA irradiance) [W/m^2/um]
    I_d_lamb
      Direct normal irradiance (direct irradiance on a horizontal surface) [W/m^2/um]
    T_a_lamb
      Transmission due to aerosols
    T_w_lamb
      Transmission due to water vapor
    T_o_lamb
      Transmission due to ozone
    T_u_lamb
      Transmission due to mixed gas (N2+O2+Ar)
    """

    # Ensure input array is sorted for calculation
    # Restore input order upon function return
    sortIndices = np.argsort(lambdaVec)
    restoreIndices = np.argsort(sortIndices)
    lambdaVec = lambdaVec[sortIndices]
    # --Create range of wavelengths within AERONET wavelength range--
    lambnew = lambdaVec * 1e6  # convert from [m] to [μm]
    if np.min(lambnew) < 0.3:
        raise ValueError('Minimum value below AERONET wavelength range')
    if np.max(lambnew) > 1.64:
        raise ValueError('Maximum value above AERONET wavelength range')

    dayOfYear = date.timetuple().tm_yday
    solarZenith = np.radians(90 - get_altitude(lat, lon, date))
    # Consider it nighttime when the solar disc is below the refracted
    # horizon (>91°).  This code does not handle twilight conditions.
    # Note also that the site elevation is assumed to be 0, so that the
    # geometric horizon is 90°.
    isNight = solarZenith > 1.5885
    cosSolZen = np.cos(solarZenith)
    sinSolZen = np.sin(solarZenith)
    solarAzimuth = np.radians(get_azimuth(lat, lon, date))
    print('Calculated Solar Zenith Angle = '
          + '{:g} degrees'.format(np.degrees(solarZenith)))
    print('Calculated Solar Azimuth Angle = '
          + '{:g} degrees'.format(np.degrees(solarAzimuth)))

    surfTilt = np.radians(surfaceTilt)
    surfDir = np.radians(surfaceDirection)
    
    #Print warning when tilt and direction are within +/-10degrees of solar zenith and azimuth
    tenRads = np.radians(10)
    isDirect = np.round(surfTilt,2) >= np.round(solarZenith-tenRads,2) and np.round(surfTilt,2) <= np.round(solarZenith+tenRads,2) \
    and np.round(surfDir,2) >= np.round(solarAzimuth-tenRads, 2) and np.round(surfDir,2) <= np.round(solarAzimuth+tenRads,2)
    if isDirect:
        print("***WARNING: Tilt and Direction are within +/-10 degrees of solar zenith and azimuth, \n"
              +"model neglects direct radiance component thus will underestimate total sky radiance***")

    # --atmospherics--
    pWVcm = precipWater * 2.54  # [inches] to [cm]
    O3 = ozone / 1000  # [DU] to [atm-cm]

    # --Read-in exo-atmospheric irradiance values from BR1986--
    sys.path.insert(0, parent_dir)
    df = pd.read_excel(parent_dir + '/Standard.xlsx')
    stan = df.values
    lamb = stan[:, 0]  # from BR1986 Table 1
    irradiances = stan[:, 1]

    x = [0.340, 0.380, 0.440, 0.500, 0.675, 0.870, 1.020, 1.640]
    AODinterp = interp1d(x, AODarray, kind="cubic", fill_value="extrapolate")
    AOD = AODinterp(lambnew)

    # --Direct Normal Irradiance Calcation--
    # the direct irradiance on a surface normal to the direction
    # of the sun at ground level for wavelength lamb
    H_0_lamb = irradiances
    f1 = interp1d(lamb, H_0_lamb, kind="cubic", fill_value="extrapolate",
                  bounds_error=False)
    H_0_lamb = f1(lambnew)

    # --Sun Earth distance factor (D)--
    phi = 2 * np.pi * (dayOfYear - 1) / 365
    D = (1.00011 + 0.034221 * np.cos(phi) + 0.00128 * np.sin(phi)
         + 0.000719 * np.cos(2 * phi) + 0.000077 * np.sin(2 * phi))

    # --Rayleigh scattering transmittance (T_r_lamb) [eqs (2-4),(2-5)]--
    P_0 = 1013  # standard pressure [mbar]
    # Source paper [Kasten 1966] does not attempt to define this term
    # beyond solar zenith of 90 degrees.  Since maximum angle at which
    # the solar disc is visible, accounting for refraction, is a zenith
    # of ~90.85°, and the Kasten function peaks at 91° (1.5885 rad)
    # (and the air mass going down for larger values is nonsensical),
    # cap the zenith value when computing the air mass.
    M = 1 / (cosSolZen + 0.15 * (93.885 - np.degrees(solarZenith)) ** (-1.253))
    M_prime = M * pressure / P_0
    T_r_lamb = np.exp(-M_prime / (lambnew ** 4 * (115.6406 - 1.335 / lambnew ** 2)))

    # --Aerosol scattering and absorption transmittance function (T_a_lamb)--
    a1Ind = np.searchsorted(lambnew, 0.5)
    a1 = np.ones(a1Ind) * 1.0274  # a1 = 1.0274 for wavelengths <0.5
    a2 = np.ones(len(lambnew) - a1Ind) * 1.2060  # a2 = 1.2060 for wavelengths >0.5
    alpha = np.append(a1, a2)
    tau500 = AOD[a1Ind] * (lambnew / 0.5) ** (-alpha)
    T_a_lamb = np.exp(-tau500 * M)
    # beta_500  = 0.1
    # T_a_lamb  = np.exp(-beta_500 * M * lambnew ** alpha)

    # --Water vapor absorption transmittance function (T_w_lamb)--
    a_w = stan[:, 2]  # BR1986 Table 1
    f2 = interp1d(lamb, a_w, kind="linear", fill_value="extrapolate",
                  bounds_error=False)
    a_w = f2(lambnew)
    T_w_lamb = np.exp(-0.2385 * a_w * pWVcm * M
                      / (1 + 20.07 * a_w * M * pWVcm) ** 0.45
                      )

    # Ozone absorption transmittance function (T_o_lamb)
    a_o = stan[:, 3]  # BR1986 Table 1
    f3 = interp1d(lamb, a_o, kind="linear", fill_value="extrapolate",
                  bounds_error=False)
    a_o = f3(lambnew)
    h0Scaled = ozonePeakHeight / 6370
    # TODO: Allow h_o to change based on longitude, latitude, and dayOfYear
    M_o = (1 + h0Scaled) / np.sqrt(cosSolZen ** 2 + 2 * h0Scaled)  # ozone mass
    T_o_lamb = np.exp(-a_o * O3 * M_o)

    # --Uniformly mixed gas absorption transmittance function (T_u_lamb)--
    a_u = stan[:, 4]  # BR1986 Table 1
    f4 = interp1d(lamb, a_u, kind="linear", fill_value="extrapolate",
                  bounds_error=False)
    a_u = f4(lambnew)
    T_u_lamb = np.exp(-1.41 * a_u * M_prime / (1 + 118.3 * a_u * M_prime) ** 0.45)

    if isNight:
        # Nothing more to do, return
        L = np.zeros(len(lambdaVec))
        L_s = np.zeros(len(lambdaVec))
        Irrad = np.zeros(len(lambdaVec))
        Irrad_tilt = np.zeros(len(lambdaVec))
        I_d_lamb = np.zeros(len(lambdaVec))
        return lambnew, L, L_s, Irrad, Irrad_tilt, lamb, irradiances, I_d_lamb,  T_a_lamb, T_w_lamb, T_o_lamb, T_u_lamb


    # --Calculate direct normal irradiance--
    I_d_lamb = H_0_lamb * D * T_r_lamb * T_a_lamb * T_w_lamb * T_o_lamb * T_u_lamb

    # --Direct Irradiance on Horizontal Surface--
    # I_d_horiz_lamb = I_d_lamb * cosSolZen

    # --Diffuse Irradiance Calculation--
    omeg = 0.945
    omegp = 0.095
    omegL = omeg * np.exp(-omegp * np.log(lambnew / 0.4) ** 2)  # Calculate wavelength-dependent single-scattering albedo (taken from NREL SPCTRAL2_PC Fortran code)
    T_aa_lamb = np.exp(-(1 - omegL) * tau500 * M)
    T_as_lamb = np.exp(-omegL * tau500 * M)
    ALG = np.log(1 - g)
    AFS = ALG * (1.459 + ALG * (0.1595 + ALG * 0.4129))
    BFS = ALG * (0.0783 + ALG * (-0.3824 - ALG * 0.5874))
    F_s = 1 - 0.5 * np.exp((AFS + BFS * cosSolZen) * cosSolZen)

    # --Calculate Rayleigh scattering component--
    I_r_lamb = H_0_lamb * D * cosSolZen * T_o_lamb * T_u_lamb * T_w_lamb * T_aa_lamb * (1 - (T_r_lamb ** 0.95)) * 0.5

    # --Calculate Aerosol scattering component--
    I_a_lamb = H_0_lamb * D * cosSolZen * T_o_lamb * T_u_lamb * T_w_lamb * T_aa_lamb * (T_r_lamb ** 1.5) * (1 - T_as_lamb) * F_s

    # Calculate sky reflectivity (r_s_lamb)
    # Primed values where M = 1.8
    M1 = 1.8
    T_o_p = np.exp(-a_o * O3 * M1)
    T_w_p = np.exp(-0.2385 * a_w * pWVcm * M1 / (1 + 20.07 * a_w * M1 * pWVcm) ** 0.45)
    T_aa_p = np.exp(-(1 - omegL) * tau500 * M1)
    T_as_p = np.exp(-omegL * tau500 * M1)
    M1_prime = M1 * pressure / P_0
    T_r_p = np.exp(-M1_prime / (lambnew ** 4 * (115.6406 - 1.335 / lambnew ** 2)))
    F_s_p = 1 - 0.5 * np.exp((AFS + BFS / 1.8) / 1.8)
    r_s_lamb = T_o_p * T_w_p * T_aa_p * (0.5 * (1 - T_r_p) + (1 - F_s_p) * T_r_p * (1 - T_as_p))
    # r_g_lamb is the ground albedo, which should be a function of wavelength,
    # the BR1986 model is unclear on how to calculate this and just uses a single value (0.2) for the values in Table 2
    # TODO: come up with a ground-albedo vector that varies with wavelength

    # --Calculate ground reflection component--
    I_g_lamb = (I_d_lamb * cosSolZen + I_r_lamb + I_a_lamb) * r_s_lamb * r_g_lamb / (1 - r_s_lamb * r_g_lamb)

    # --Calculate diffuse irradiance on a horizontal surface--
    ind45 = np.searchsorted(lambnew, 0.45)
    C_s_a = (lambnew[0:ind45] + 0.55) ** 1.8  # Cs = (lambda + 0.55)^1.8 for lambda < 0.45
    one = np.ones(len(lambnew[ind45:]))   # Cs = 1.0 for lambda > 0.45
    C_s = np.append(C_s_a, one)

    I_s_lamb = C_s * (I_r_lamb + I_a_lamb + I_g_lamb)

    # --Diffuse irradiance on an inclined plane calculation--
    # calculate angle of incidence in 3D plane

    surfNorm = [np.cos(surfDir) * np.sin(surfTilt),
                np.sin(surfDir) * np.sin(surfTilt),
                np.cos(surfTilt)]

    solVec = [np.cos(solarAzimuth) * sinSolZen,
              np.sin(solarAzimuth) * sinSolZen,
              cosSolZen]

    cosTheta = np.clip(np.dot(surfNorm, solVec), 0, 1)
    # Clamp range to [0, 1], because no flux flows from the back
    # of the surface to the front

    # --Total solar irradiance on a horizontal surface--
    I_T_lamb = I_d_lamb * cosSolZen + I_s_lamb  # same as 0 degree tilt

    I_T_tilt_lamb = (I_d_lamb * cosTheta
                     + I_s_lamb * ((I_d_lamb * cosTheta
                                    / (H_0_lamb * D * cosSolZen)
                                    )
                                   + 0.5 * ((1 + np.cos(surfTilt))
                                            * (1 - I_d_lamb / (H_0_lamb * D))
                                            )
                                   )
                     + 0.5 * I_T_lamb * r_g_lamb * (1 - np.cos(surfTilt))
                     )

    # --Solar spectral RADIANCE Calculation--
    # HemisphereSolidAngle = 2*spc.pi  # convert to sky spectral radiance
    # L = I_T_tilt_lamb/HemisphereSolidAngle # sky spectral radiance (units: W*m^-2*micron^-1*sr^-1)
    L = I_T_tilt_lamb / spc.pi
    L *= 1E6  # convert output from units of W*m^-2*micron^-1*sr^-1 so its in units of W*m^-2*m^-1*sr^-1
    L_s = I_s_lamb /(spc.pi) #Calculate radiance from just the diffuse scatter component
    L_s *= 1E6

    # L is the radiance calculated from tilted irradiance, irradiances are TOA irradiances
    Irrad = I_T_lamb
    Irrad_tilt = I_T_tilt_lamb

    # Restore ordering with respect to input lambdaVec
    lambnew = lambnew[restoreIndices]
    L = L[restoreIndices]
    Irrad = Irrad[restoreIndices]
    Irrad_tilt = Irrad_tilt[restoreIndices]
    I_d_lamb = I_d_lamb[restoreIndices]
    return lambnew, L, L_s, Irrad, Irrad_tilt, lamb, irradiances, I_d_lamb, T_a_lamb, T_w_lamb, T_o_lamb, T_u_lamb


def makeWaterLines(lambdaL, T):
    # Creates Rotational-Vibrational lines for H20
    # Data and process from G. Avila, J.M. Fernandez, B. Mate, G. Tejeda, and S. Montero
    # "Ro-vibrational Raman Cross SEctions of Water Vapor in the OH Stretching Region"
    # Journal of Molecular Spectroscopy, Vol. 196, pp. 77-92, 1999
    # Inputs
    #   E_i     Energy (in cm-1) of the initial level.
    #   J_i   Rotational quantum number initial state.
    #   Ka_i  Rotational quantum number initial state.
    #   Kc_i  Rotational quantum number initial state.
    #   vib_i      Vibrational quantum numbers of initial state (ground state).
    #   E_f      Energy (in cm-1) of the final level.
    #   J_f   Rotational quantum number final state.
    #   Ka_f  Rotational quantum number final state.
    #   Kc_f  Rotational quantum number final state.
    #   vib_f    Vibrational quantum numbers of the final state.
    #   nu     Wavenumber (in cm-1) of the transition.
    #   Axx Coefficients A[XX]  of Eq. (29), in m^6.
    #   Axy Coefficients A[XY] of Eq. (29), in m^6.
    # Outputs
    #   dLambdaSort - output vector of wavelengths
    #   dRCSSort - output vector of differential backscatter cross section (m^2/sr)

    kb = 0.695034800  # Boltzmann's constant in units of 1/cm/K

    # Data available to download on the website as supplemental
    f = open('./Avila1999_H20_Data.txt', 'r')
    startLine = 40  # skips first 40 lines of text file

    data = f.readlines()[startLine:-2]

    E_i = np.array([])
    J_i = np.array([])
    Ka_i = np.array([])
    Kc_i = np.array([])
    vib_i = np.array([])
    E_f = np.array([])
    J_f = np.array([])
    Ka_f = np.array([])
    Kc_f = np.array([])
    vib_f = np.array([])
    nu = np.array([])
    Axx = np.array([])
    Axy = np.array([])

    for line in data:
        words = line.split(";")
        E_i = np.append(E_i, float(words[0]))
        J_i = np.append(J_i, int(words[1]))
        Ka_i = np.append(Ka_i, int(words[2]))
        Kc_i = np.append(Kc_i, int(words[3]))
        vib_i = np.append(vib_i, int(words[4]))
        E_f = np.append(E_f, float(words[5]))
        J_f = np.append(J_f, int(words[6]))
        Ka_f = np.append(Ka_f, int(words[7]))
        Kc_f = np.append(Kc_f, int(words[8]))
        vib_f = np.append(vib_f, int(words[9]))
        nu = np.append(nu, float(words[10]))
        Axx = np.append(Axx, float(words[11]))
        Axy = np.append(Axy, float(words[12]))

    # Table 7 in publication
    # Temperature in Kelvin
    T_rovib = np.array([0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275, 295, 300, 325, 350, 375, 400, 425, 450, 475, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    # Partition function Z(T)
    ZT_rovib = np.array([1, 1, 1.002, 1.01, 1.031, 1.07, 1.132, 1.218, 1.327, 1.611, 1.968, 2.384, 2.847,
                         3.345, 4.706, 6.185, 7.755, 9.407, 1.114E1, 1.295E1, 1.68E1, 2.096E1, 2.54E1, 3.013E1, 3.511E1, 4.865E1,
                         6.361E1, 7.985E1, 9.73E1, 1.159E2, 1.355E2, 1.562E2, 1.735E2, 1.779E2, 2.006E2, 2.242E2, 2.488E2, 2.743E2, 3.008E2,
                         3.282E2, 3.566E2, 3.860E2, 4.479E2, 5.139E2, 5.841E2, 6.588E2, 7.379E2, 8.217E2, 9.102E2, 1.003E3, 1.101E3, 1.204E3])
    f = interp1d(T_rovib, ZT_rovib)

    nu0 = 1 / lambdaL

    dlambda = 1 / (nu0 - nu * 100)

    Ztfit = f(T)

    dRCSout = np.zeros([np.size(E_i), np.size(T)])

    # TODO: This part of the code runs slow as its contains 7081 Raman shifts for each altitude/
    # temperature value. Should look in the future at methods of decreasing run time.
    # Potentially by removing Ro-Vibe shifts far from the water line and/or selecting
    # only the top ~95% or so contributors to the cross section.
    for i in np.arange(0, np.size(Ztfit), 1):
        # Equation 29 from Avila
        # Axx is parallel polarization
        # Axy is perpendicular poloarization
        # nu isin units of 1/cm so need to multiply by 100 to get 1/m
        # note that kb is Boltzmann's constant in units of 1/cm/K so no unit conversion is required
        dRCSxx = (nu0 - nu * 100) ** 4 * np.exp(-E_i / (kb * T[i])) / Ztfit[i] * Axx
        dRCSxy = (nu0 - nu * 100) ** 4 * np.exp(-E_i / (kb * T[i])) / Ztfit[i] * Axy
        # TODO: Do we need both xx and yy?
        dRCS = dRCSxx + dRCSxy

        indices = dlambda.argsort()  # sort array for numerical, increasing order
        dlambdaSort = dlambda[indices]  # sort both arrays by increasing lambda
        dRCSSort = dRCS[indices]
        # nuSort = np.flip(nu[indices])
        dRCSout[:, i] = dRCSSort
        # dRCSout = np.transpose(dRCSout)

    return [dlambdaSort, dRCSout]


def makeVibrationalLines(numLines, gj_even, gj_odd, B0, B1, I, lambdaL, T, nuVib, gammaPSquared, aPSquared):
    # Generates Vibrational-Rotational Raman lines
    # Equations from Chapter 9 of Lidar: Range-Resolved Optical Remote
    # Sensing of the Atmosphere by Claus Weitkemp
    # And also from Liu and Yi 2014
    # Inputs
    #   numLines - maximum number of frequency lines to consider (number - unitless)
    #     less than or equal to 22 per Liu and Yi
    #   gj_even - statistical weight factor for even rotational quantum state J (unitless)
    #   gj_odd - statistical weight factor for odd rotational quantum state J (unitless)
    #   B0 - ground-state rotational distortion constant (m^-1)
    #   B1 - ground-state rotational distortion constant (m^-1)
    #   I - nuclear spin
    #   lambdaL - excitation wavelength (meters)
    #   gammaPSquared - change of polarizability with changing distance of the square of the  anisotropy of the molecular polarizability tensor (m^6)
    #   aPSquared - change of polarizability with changing distance of the square of the mean polarizability of the molecular polarizability tensor (m^6)
    #   T - temperature (Kelvin)
    #   nuVib - vibrational wavenumber for species AKA oscillator frequency of the molecule (1/cm)
    # Outputs
    #   dlambda - output vector of Rotational Raman wavelengths (meters)
    #   dRCS - differential backscatter cross section (m^2/sr)

    bnu = spc.h / (8 * np.pi ** 2 * spc.c * nuVib)

    # numLines-1 index made here due to how linesQ,S,O are coded
    AdlambdaQ = np.zeros(numLines - 1)
    AdlambdaS = np.zeros(numLines - 1)
    AdlambdaO = np.zeros(numLines - 1)
    AdRCS_stokes_Q = np.zeros([numLines - 1, np.size(T)])
    AdRCS_stokes_S = np.zeros([numLines - 1, np.size(T)])
    AdRCS_stokes_O = np.zeros([numLines - 1, np.size(T)])

    for i in np.arange(0, np.size(T)):
        [AdlambdaQ, AdRCS_stokes_Q[:, i]] = linesQ(numLines, lambdaL, nuVib, B0, B1, I, gj_even, gj_odd, bnu, aPSquared, gammaPSquared, T[i])
        [AdlambdaS, AdRCS_stokes_S[:, i]] = linesS(numLines, lambdaL, nuVib, B0, B1, I, gj_even, gj_odd, bnu, aPSquared, gammaPSquared, T[i])
        [AdlambdaO, AdRCS_stokes_O[:, i]] = linesO(numLines, lambdaL, nuVib, B0, B1, I, gj_even, gj_odd, bnu, aPSquared, gammaPSquared, T[i])

    dlambda = np.concatenate([np.flip(AdlambdaO), np.flip(AdlambdaQ), AdlambdaS])
    # Need a axis=0 to ensure temperature columns aren't flipped and rather values for a single temperature
    dRCS = np.concatenate([np.flip(AdRCS_stokes_O, axis=0), np.flip(AdRCS_stokes_Q, axis=0), AdRCS_stokes_S])

    return [dlambda, dRCS]


def linesQ(numLines, lambdaL, nu_Vib, B0, B1, Ix, gj_even, gj_odd, bnuS, aPSquared, gammaPSquared, T):
    # Generates Vibrational-Rotational Raman lines of the Q branch
    # Equations from Chapter 9 of Lidar: Range-Resolved Optical Remote
    # Sensing of the Atmosphere by Claus Weitkemp
    # And also from Liu and Yi 2014

    # Inputs
    #   numLines - maximum number of frequency lines to consider (number - unitless)
    #   gj_even - statistical weight factor for even rotational quantum state J (unitless)
    #   gj_odd - statistical weight factor for odd rotational quantum state J (unitless)
    #   B0 - ground-state rotational distortion constant (m^-1)
    #   B1 - ground-state rotational distortion constant (m^-1)
    #   I - nuclear spin
    #   lambdaL - excitation wavelength (meters)
    #   gammaPSquared - change of polarizability with changing distance of the square of the  anisotropy of the molecular polarizability tensor (m^6)
    #   aPSquared - change of polarizability with changing distance of the square of the mean polarizability of the molecular polarizability tensor (m^6)
    #   T - temperature (Kelvin)
    #   nuVib - vibrational wavenumber for species AKA oscillator frequency of the molecule (1/cm)
    #   bnuS - zero point amplitude in vibrational mode
    # Outputs
    #   dlambdaQ - output vector of Vibrational Raman wavelengths (meters)
    #   dRCS_stokes_Q - differential backscatter cross section of this branch (m^2/sr)

    dnu = np.array([])
    PhiJ_StokesVR_Qbranch = np.array([])
    dRCS_stokes_Q = np.array([])

    nu0 = 1 / lambdaL
    k_nu = (2 * np.pi) ** 4
    Q = spc.Boltzmann * T / (2 * spc.h * spc.c * B0)

    for i in range(1, numLines):
        # J = 0,1,2...
        J = i - 1

        if (J % 2 == 0):
            gj = gj_even
        else:
            gj = gj_odd

        # Eq 2.2
        dnu = np.append(dnu, nu_Vib + J * (J + 1) * (B1 - B0))

        # Eq 3.2
        # Liu and Yi use (J+1) in place of J as a multiplier of gammaPSquared...why?
        PhiJ_StokesVR_Qbranch = np.append(PhiJ_StokesVR_Qbranch, bnuS * (2 * J + 1) / (1 - np.exp(-spc.h * spc.c * nu_Vib / (spc.Boltzmann * T))) * (aPSquared + 7 * J * (J + 1) * gammaPSquared / (45 * (2 * J + 3) * (2 * J - 1))))

        # Eq 1
        # Liu and Yi show and extra factor of (2I+1)**2 in the denominator...why?
        # Liu and Yi use different units for the values of gamma and a which results in changing the value of k_nu
        dRCS_stokes_Q = np.append(dRCS_stokes_Q, k_nu * (nu0 - dnu[J]) ** 4 * gj * PhiJ_StokesVR_Qbranch[J] / (Q * (2 * Ix + 1) ** 2) * np.exp(-B0 * spc.h * spc.c * J * (J + 1) / (spc.Boltzmann * T)))

    dlambdaQ = 1 / (nu0 - dnu)

    return [dlambdaQ, dRCS_stokes_Q]


def linesS(numLines, lambdaL, nu_Vib, B0, B1, Ix, gj_even, gj_odd, bnuS, aPSquared, gammaPSquared, T):
    # Generates Vibrational-Rotational Raman lines of the S branch
    # Equations from Chapter 9 of Lidar: Range-Resolved Optical Remote
    # Sensing of the Atmosphere by Claus Weitkemp
    # And also from Liu and Yi 2014

    # Inputs
    #   numLines - maximum number of frequency lines to consider (number - unitless)
    #   gj_even - statistical weight factor for even rotational quantum state J (unitless)
    #   gj_odd - statistical weight factor for odd rotational quantum state J (unitless)
    #   B0 - ground-state rotational distortion constant (m^-1)
    #   B1 - ground-state rotational distortion constant (m^-1)
    #   I - nuclear spin
    #   lambdaL - excitation wavelength (meters)
    #   gammaPSquared - change of polarizability with changing distance of the square of the  anisotropy of the molecular polarizability tensor (m^6)
    #   aPSquared - change of polarizability with changing distance of the square of the mean polarizability of the molecular polarizability tensor (m^6)
    #   T - temperature (Kelvin)
    #   nuVib - vibrational wavenumber for species AKA oscillator frequency of the molecule (1/cm)
    #   bnuS - zero point amplitude in vibrational mode
    # Outputs
    #   dlambdaS - output vector of Vibrational Raman wavelengths (meters)
    #   dRCS_stokes_S - differential backscatter cross section of this branch (m^2/sr)

    dnu = np.array([])
    PhiJ_StokesVR_Sbranch = np.array([])
    dRCS_stokes_S = np.array([])

    nu0 = 1 / lambdaL
    k_nu = (2 * np.pi) ** 4
    Q = spc.Boltzmann * T / (2 * spc.h * spc.c * B0)

    for i in range(1, numLines):
        # J = 0,1,2...
        J = i - 1

        if (J % 2 == 0):
            gj = gj_even
        else:
            gj = gj_odd

        # Eq 2.1
        dnu = np.append(dnu, nu_Vib + (4 * J + 6) * B1)

        # Eq 3.1
        PhiJ_StokesVR_Sbranch = np.append(PhiJ_StokesVR_Sbranch, bnuS / (1 - np.exp(-spc.h * spc.c * nu_Vib / (spc.Boltzmann * T))) * (7 * (J + 1) * (J + 2) * gammaPSquared / (30 * (2 * J + 3))))

        # Eq 1
        # Liu and Yi show and extra factor of (2I+1)**2 in the denominator...why?
        # Liu and Yi use different units for the values of gamma and a which results in changing the value of k_nu
        dRCS_stokes_S = np.append(dRCS_stokes_S, k_nu * (nu0 - dnu[J]) ** 4 * gj * PhiJ_StokesVR_Sbranch[J] / (Q * (2 * Ix + 1) ** 2) * np.exp(-B0 * spc.h * spc.c * J * (J + 1) / (spc.Boltzmann * T)))

    dlambdaS = 1 / (nu0 - dnu)

    return [dlambdaS, dRCS_stokes_S]


def linesO(numLines, lambdaL, nu_Vib, B0, B1, Ix, gj_even, gj_odd, bnuS, aPSquared, gammaPSquared, T):
    # Generates Vibrational-Rotational Raman lines of the O branch
    # Equations from Chapter 9 of Lidar: Range-Resolved Optical Remote
    # Sensing of the Atmosphere by Claus Weitkemp
    # And also from Liu and Yi 2014
    # Inputs
    #   numLines - maximum number of frequency lines to consider (number - unitless)
    #   gj_even - statistical weight factor for even rotational quantum state J (unitless)
    #   gj_odd - statistical weight factor for odd rotational quantum state J (unitless)
    #   B0 - ground-state rotational distortion constant (m^-1)
    #   B1 - ground-state rotational distortion constant (m^-1)
    #   I - nuclear spin
    #   lambdaL - excitation wavelength (meters)
    #   gammaPSquared - change of polarizability with changing distance of the square of the  anisotropy of the molecular polarizability tensor (m^6)
    #   aPSquared - change of polarizability with changing distance of the square of the mean polarizability of the molecular polarizability tensor (m^6)
    #   T - temperature (Kelvin)
    #   nuVib - vibrational wavenumber for species AKA oscillator frequency of the molecule (1/cm)
    #   bnuS - zero point amplitude in vibrational mode
    # Outputs
    #   dlambdaO - output vector of Vibrational Raman wavelengths (meters)
    #   dRCS_stokes_O - differential backscatter cross section of this branch (m^2/sr)
    dnu = np.array([])
    PhiJ_StokesVR_Obranch = np.array([])
    dRCS_stokes_O = np.array([])

    nu0 = 1 / lambdaL
    k_nu = (2 * np.pi) ** 4
    Q = spc.Boltzmann * T / (2 * spc.h * spc.c * B0)

    for i in range(1, numLines):
        # J=2,3,4...
        J = i + 1

        if (J % 2 == 0):
            gj = gj_even
        else:
            gj = gj_odd

        # Eq 2.3
        dnu = np.append(dnu, nu_Vib - (4 * J - 2) * B0)

        # Eq 3.3
        PhiJ_StokesVR_Obranch = np.append(PhiJ_StokesVR_Obranch, bnuS / (1 - np.exp(-spc.h * spc.c * nu_Vib / (spc.Boltzmann * T))) * (7 * J * (J - 1) * gammaPSquared / (30 * (2 * J - 1))))

        # Eq 1
        # Liu and Yi show and extra factor of (2I+1)**2 in the denominator...why?
        # Liu and Yi use different units for the values of gamma and a which results in changing the value of k_nu
        dRCS_stokes_O = np.append(dRCS_stokes_O, k_nu * (nu0 - dnu[J - 2]) ** 4 * gj * PhiJ_StokesVR_Obranch[J - 2] / (Q * (2 * Ix + 1) ** 2) * np.exp(-B0 * spc.h * spc.c * J * (J + 1) / (spc.Boltzmann * T)))

    dlambdaO = 1 / (nu0 - dnu)

    return [dlambdaO, dRCS_stokes_O]


def makeRotationalLines(NumLines, gj_even, gj_odd, B0, D0, I, lambdaL, gammaSquared, T):
    # Generates Rotational Raman lines
    # Equations from Chapter 10 of Lidar: Range-Resolved Optical Remote
    # Sensing of the Atmosphere by Claus Weitkemp
    # Inputs
    #   NumLines - maximum number of frequency lines to consider (number - unitless)
    #   gj_even - statistical weight factor for even rotational quantum state J (unitless)
    #   gj_odd - statistical weight factor for odd rotational quantum state J (unitless)
    #   B0 - ground-state rotational distortion constant (m^-1)
    #   D0 - ground-state distortional constant (m^-1)
    #   I - nuclear spin
    #   lambdaL - excitation wavelength (meters)
    #   gammaSquared - square of the anisotropy of the molecular polarizability tensor (m^6)
    #   T - temperature (Kelvin)
    # Outputs
    #   dlambda - output vector of Rotational Raman wavelengths (meters)
    #   dRCS - differential backscatter cross section (m^2/sr)

    dnu_Stokes = np.array([])
    i_Erot = np.array([])
    Xj_Stokes = np.array([])
    dRcs_Stokes = []
    anti_dnu_Stokes = np.array([])
    anti_Xj_Stokes = np.array([])
    anti_dRcs_Stokes = []

    nu0 = 1 / lambdaL  # wavelength to frequency conversion

    for J in range(0, NumLines + 1):
        i_Erot = np.append(i_Erot, ((B0 * J * (J + 1) - D0 * (J ** 2) * ((J + 1) ** 2))) * spc.h * spc.speed_of_light)

    for i in range(1, NumLines):
        J = i - 1  # For Stokes Branch, J=0,1,2,3,4...
        if (J % 2 == 0):
            gj = gj_even
        else:
            gj = gj_odd

        # Erot= np.append(Erot,i_Erot)

        i_dnu_Stokes = -B0 * 2 * (2 * J + 3) + D0 * (3 * (2 * J + 3) + (2 * J + 3) ** 3)
        dnu_Stokes = np.append(dnu_Stokes, i_dnu_Stokes)
        i_Xj_Stokes = (J + 1.0) * (J + 2) / (2 * J + 3)
        Xj_Stokes = np.append(Xj_Stokes, i_Xj_Stokes)

        dRcs_Stokes.append((122.0 * ((np.pi) ** 4) / 15) * (((gj * spc.h * spc.speed_of_light * B0 * ((nu0 + i_dnu_Stokes) ** 4)) * gammaSquared) / (((2 * I + 1) ** 2) * spc.k * T)) * i_Xj_Stokes * np.exp(-i_Erot[J] / (spc.k * T)))

        J = i + 1  # For Anti-Stokes Branch, J=2,3,4...
        if (J % 2 == 0):
            gj = gj_even
        else:
            gj = gj_odd

        i_anti_dnu_Stokes = B0 * 2 * (2 * J - 1) - D0 * (3 * (2 * J - 1) + (2 * J - 1) ** 3)
        anti_dnu_Stokes = np.append(anti_dnu_Stokes, i_anti_dnu_Stokes)
        i_anti_Xj_Stokes = J * (J - 1.0) / (2 * J - 1)
        anti_Xj_Stokes = np.append(anti_Xj_Stokes, i_anti_Xj_Stokes)

        anti_dRcs_Stokes.append((122.0 * ((np.pi) ** 4) / 15) * (((gj * spc.h * spc.speed_of_light * B0 * ((nu0 + i_anti_dnu_Stokes) ** 4)) * gammaSquared) / (((2 * I + 1) ** 2) * spc.k * T)) * i_anti_Xj_Stokes * np.exp(-i_Erot[J] / (spc.k * T)))

    dRcs_Stokes = np.array(dRcs_Stokes)
    anti_dRcs_Stokes = np.array(anti_dRcs_Stokes)

    # Combine stokes and anti-stokes into a single vector
    dnu = np.concatenate([np.flip(anti_dnu_Stokes), dnu_Stokes])
    dRCS = np.concatenate([np.flip(anti_dRcs_Stokes,0), dRcs_Stokes])

    dlambda = 1 / (dnu + nu0)

    return [dlambda, dRCS]


    #TODO: Add acceleration due to gravity calculation? (Can pull from USU code)
    
def HV57(h,W,A):
    # This function creates a  Hufnagel Valley model turbulence profile
    #Inputs
    # h altitude vector in meters
    # W - the root mean squared wind speed over the 5-20 km altitude range (m/s) - typically set to 21
    # A - ground-level Cn2 value (m^-2/3) - set to 1.7E-14
    #Outputs
    # Cn2 is the Refractive Index Structure Constant (m^-2/3)
    h=h/1000 #Convert m to km as HV model requires km
    
    Cn2 = 8.2E-26*(W**2)*(h**10)*np.exp(-h)+ 2.7E-16*np.exp(-h/1.5) + A * np.exp(-h/.1)
    return Cn2

    