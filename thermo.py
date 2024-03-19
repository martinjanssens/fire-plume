import numpy as np
import xarray as xr
import scipy.optimize

# Python version of the thermodynamic calculations in DALES (no ice)

# constants as in DALES
rd      = 287.04          # gas constant for dry air.
rv      = 461.5           # gas constant for water vapor.
cp      = 1004.           # specific heat at constant pressure (dry air).
rlv     = 2.53e6          # latent heat for vaporisation
pref0   = 1.e5            # standard pressure used in exner function.
grav    = 9.81            # gravity acceleration.

# Saturation water pressure over liquid
# as in DALES
def esatl(T):
     return np.exp(54.842763-6763.22/T-4.21*np.log(T)+0.000367*T+
         np.tanh(0.0415*(T-218.8))*(53.878-1331.22/T-9.44523*np.log(T)+ 0.014025*T))

# exner function of pressure
def exnf(pres):
    return (pres/pref0)**(rd/cp)

# saturation humidity as in DALES
def qsatur(T, pres):
    esl1 = esatl(T)

    # this breaks if vapor pressure > real pressure
    # i.e. above boiling point of water at current pressure
    # with clamp
    qsat  = (rd/rv)*esl1 / (pres - np.minimum( (1.-rd/rv)*esl1, pres*0.8) )
    return np.clip(qsat, 0, 0.9)

# get (T, q_l) from (theta_l, q_t) at pressure pres
# We only have a scalar version, for clarity
def calculate_T_q_l(theta_l, q_t, pres):
        
    T = exnf(pres) * theta_l # first guess
    qsat = qsatur(T, pres)

    def theta_l_err(t):
        q_l = np.maximum(q_t - qsatur(t, pres), 0.)
        theta_l1 = t/exnf(pres) - (rlv/(cp*exnf(pres))) * q_l
        return theta_l - theta_l1

    def theta_l_err_scalar(t, q_t, pres, theta_l):
        q_l = np.maximum(q_t - qsatur(t, pres), 0.)
        theta_l1 = t/exnf(pres) - (rlv/(cp*exnf(pres))) * q_l
        return theta_l - theta_l1

    # if type(theta_l)==np.ndarray or type(theta_l)==xr.DataArray:
    #     T = np.zeros(theta_l.size)
    #     for i in range(theta_l.size):
    #         try:
    #             T[i] = scipy.optimize.brentq(theta_l_err_scalar, 200, 330, xtol=1e-3, args = (q_t[i], pres[i], theta_l[i]))
    #         except:
    #             # print('T, q_l calculation failed, assigning nan')
    #             T[i] = np.nan
    # else:
    T = scipy.optimize.brentq(theta_l_err_scalar, 200, 373.15, xtol=1e-3, args = (q_t, pres, theta_l))
    q_l = np.maximum(q_t - qsatur(T, pres), 0.) # Saturation adjustment hypothesis
    
    return T, q_l

# base pressure profile, if it's not specified
# standard atmospheric lapse rate with a surface temperature offset
# for now this version is valid only below 11 km
def calculate_pressure(zf, p_s=101300, theta_l_s=300):
    # zmat=(/11000.,20000.,32000.,47000./)           # heights of lapse rate table
    lapserate=[-6.5/1000., 0., 1./1000, 2.8/1000 ]   # lapse rate table
        
    tsurf=theta_l_s*(p_s/pref0)**(rd/cp) # surface temperature
    zsurf = 0
    #pmat = np.exp((log(ps)*lapserate(1)*rd+np.log(tsurf+zsurf*lapserate(1))*grav-
    #               np.log(tsurf+zmat(1)*lapserate(1))*grav)/(lapserate(1)*rd))

    pb = np.exp((np.log(p_s)*lapserate[0]*rd + np.log(tsurf+zsurf*lapserate[0])*grav-
                 np.log(tsurf+zf*lapserate[0])*grav)/(lapserate[0]*rd))
    
    return pb

def calculate_theta(T, pres):
    return T/exnf(pres)

# Get theta_v from theta_l and q_t (that is all the user needs to see)
def calculate_thermo(theta_l, q_t, pres):

    # Convert to floats, in case xarray values are passed
    theta_l = float(theta_l)
    q_t = float(q_t)
    pres = float(pres)

    T, q_l = calculate_T_q_l(theta_l, q_t, pres)
    theta = calculate_theta(T, pres)
    theta_v = theta * (1 - (1 - rv/rd) * q_t - rv/rd*q_l)
    return T, q_l, theta, theta_v
