import xarray as xr
import numpy as np
import thermo

def plume_ascent(theta_l_e, q_t_e, pres_e, height, theta_l_p_0, q_t_p_0, area_plume_0, w_p_0,
                 a_w = 1, b_w = 0, fac_ent = 1, beta = 1):
    """
    Dynamically lift an entraining parcel (plume) through a specified
    environment, given an initial parcel state. Loosely based on the
    formulation by Rio et al., (2010), but with a different entrainment model.

    Parameters
    ----------
    theta_l_e:       numpy array of shape (height) containing the environmental
                     liquid-water potential temperature [K]
    q_t_e:           numpy array of shape (height) containing the environmental
                     total-water specific humidity [kg water / kg air]
    pres_e:          numpy array of shape (height) containing the environmental
                     pressure (assumed equal to the plume pressure) [Pa]
    height:          numpy array containing the sounding's vertical coordinates
                     [m]
    theta_l_p_0:     float containing the initial plume temperature [K]
    q_t_p_0:         float containing the initial plume moisture [kg water / kg
                     air]
    area_plume_0:    float of initial (dimensional) plume area, typically
                     proportional to the active fire area [m^2]
    w_p_0:           float containing the initial plume vertical velocity (must
                     be > 0)
    a_w:             float scaling factor for the buoyancy term in the plume
                     vertical velocity equation [-]
    b_w:             float scaling factor for the drag term in the plume
                     vertical velocity equation [-]
    fac_ent:         Factor scaling the fractional entrainment
    beta:            Ratio of fractional entrainment to detrainment above the
                     surface layer. Normally expect this to be <=1.

    Returns
    -------
    ds: xarray Dataset
        Containing the vertical profiles of the thermodyanmic and dynamic state
        of both the plume and the environment. Quantities such as the LCL and
        stable injection height can be inferred from these profiles.

    """

    ## Checks that will save crashes
    # A minimum initial updraft speed is needed to not produce infinite entrainment
    if w_p_0<0.1:
        w_p_0 = 0.1
    
    
    ## Initialisation
        
    # Plume arrays in an xarray dataset
    init_prof = np.zeros(height.size)
    ds = xr.Dataset({'theta_l_e':('height', theta_l_e),
                 'q_t_e':('height', q_t_e),
                 'pres_e':('height', pres_e),
                 'theta_l_p':('height', np.copy(init_prof)),
                 'q_t_p':('height', np.copy(init_prof)),
                 'entrainment':('height', np.copy(init_prof)),
                 'detrainment':('height', np.copy(init_prof)),
                 'mass_flux_p':('height', np.copy(init_prof)),
                 'w_p':('height', np.copy(init_prof)),
                 'a_p':('height', np.copy(init_prof)),
                 'T_p':('height', np.copy(init_prof)),
                 'q_l_p':('height', np.copy(init_prof)),
                 'theta_p':('height', np.copy(init_prof)),
                 'theta_v_p':('height', np.copy(init_prof)),
                 'buoyancy_p':('height', np.copy(init_prof)),
                 'T_e':('height', np.copy(init_prof)),
                 'q_l_e':('height', np.copy(init_prof)),
                 'theta_e':('height', np.copy(init_prof)),
                 'theta_v_e':('height', np.copy(init_prof)),
                 'rho_e':('height', np.copy(init_prof)),
                },
                coords={'height':height})
    
    
    ## First model level
    # Thermodynamics (environment)
    ds['T_e'][0], ds['q_l_e'][0], ds['theta_e'][0], ds['theta_v_e'][0] = thermo.calculate_thermo(ds['theta_l_e'][0], ds['q_t_e'][0], ds['pres_e'][0])
    ds['rho_e'][0] = ds['pres_e'][0]/thermo.rd/ds['T_e'][0]

    # Thermodynamics (plume)
    ds['theta_l_p'][0] = theta_l_p_0
    ds['q_t_p'][0] = q_t_p_0
    ds['T_p'][0], ds['q_l_p'][0], ds['theta_p'][0], ds['theta_v_p'][0] = thermo.calculate_thermo(ds['theta_l_p'][0], ds['q_t_p'][0], ds['pres_e'][0])
    ds['buoyancy_p'][0] = thermo.grav / ds['theta_v_e'][0] * (ds['theta_v_p'][0] - ds['theta_v_e'][0])

    # Plume mass flux and entrainment
    ds['a_p'][0] = area_plume_0
    ds['w_p'][0] = w_p_0
    ds['mass_flux_p'][0] = ds['rho_e'][0] * ds['a_p'][0] * ds['w_p'][0]

    epsi = fac_ent*beta#/np.sqrt(ds['a_p'][0])
    delt = epsi/beta
    
    ds['entrainment'][0] = epsi*ds['mass_flux_p'][0]
    ds['detrainment'][0] = 0.
    
    ## Loop until plume top
    # Use all values at height index i-1 to calculate the values at height index i (simple forward Euler)
    dz = ds['height'].diff('height')
    for i in range(1, ds['height'].size):
        
        # Mass flux through plume
        ds['mass_flux_p'][i] = ds['mass_flux_p'][i-1] + (ds['entrainment'][i-1] - ds['detrainment'][i-1]) * dz[i-1]

        # Thermodynamics (environment)
        try:
            ds['T_e'][i], ds['q_l_e'][i], ds['theta_e'][i], ds['theta_v_e'][i] = thermo.calculate_thermo(ds['theta_l_e'][i], ds['q_t_e'][i], ds['pres_e'][i])
        except:
            print('Failed thermo env, height', ds['height'][i].data)
            return ds
        ds['rho_e'][i] = ds['pres_e'][i]/thermo.rd/ds['T_e'][i]
        
        # Thermodynamics (plume)
        ds['theta_l_p'][i] = ds['theta_l_p'][i-1] + ds['entrainment'][i-1] * (ds['theta_l_e'][i-1] - ds['theta_l_p'][i-1]) / ds['mass_flux_p'][i-1] * dz[i-1]
        ds['q_t_p'][i] = ds['q_t_p'][i-1] + ds['entrainment'][i-1] * (ds['q_t_e'][i-1] - ds['q_t_p'][i-1]) / ds['mass_flux_p'][i-1] * dz[i-1]
        try:
            ds['T_p'][i], ds['q_l_p'][i], ds['theta_p'][i], ds['theta_v_p'][i] = thermo.calculate_thermo(ds['theta_l_p'][i], ds['q_t_p'][i], ds['pres_e'][i])
        except:
            print('Failed thermo env, height', ds['height'][i].data)
            return ds
        ds['buoyancy_p'][i] = thermo.grav / ds['theta_v_e'][i] * (ds['theta_v_p'][i] - ds['theta_v_e'][i])

        # Vertical velocity
        # In the w-eq, the scaling factors for buoyancy (a_w) and entrainment (b_w) are not well-constrained,
        # see e.g. de Roode et al. (2012), Romps & Charn (2015). We leave them as free parameters to tune.

        dwp2 = 2*(a_w*ds['buoyancy_p'][i-1] - b_w*epsi*ds['w_p'][i-1]**2)*dz[i-1]
        if dwp2 + ds['w_p'][i-1]**2 < 0:
            # Kinetic energy cannot go below 0
            ds['w_p'][i] = 0
        else:
            ds['w_p'][i] = np.sqrt(dwp2 + ds['w_p'][i-1]**2)
        
        # Entrainment and detrainment
        # Siebesma & Holtslag (1996) - like everywhere, with epsi and delt fixed with height
        # This is almost certainly wrong, but we have to start somewhere
        ds['entrainment'][i] = epsi * ds['mass_flux_p'][i]
        ds['detrainment'][i] = delt * ds['mass_flux_p'][i]

        # Update area fraction to be consistent with the new mass flux, vertical velocity and density
        ds['a_p'][i] = ds['mass_flux_p'][i] / (ds['rho_e'][i] * ds['w_p'][i])

        # Discontinuous plume top where the plume is no longer ascending, or the area is zero
        if ds['w_p'][i] <= 0 or ds['a_p'][i] <= 0:

            # Above this height, complete the environmental and plume thermodynamic profiles so they are equal
            for j in range(i, ds['height'].size):

                # Environment
                ds['T_e'][j], ds['q_l_e'][j], ds['theta_e'][j], ds['theta_v_e'][j] = thermo.calculate_thermo(ds['theta_l_e'][j], ds['q_t_e'][j], ds['pres_e'][j])
                ds['rho_e'][j] = ds['pres_e'][j]/thermo.rd/ds['T_e'][j]
                ds['theta_l_p'][j] = ds['theta_l_e'][j]
                ds['q_t_p'][j] = ds['q_t_e'][j]
                ds['T_p'][j] = ds['T_e'][j]
                ds['q_l_p'][j] = ds['q_l_e'][j]
                ds['theta_p'][j] = ds['theta_e'][j]
                ds['theta_v_p'][j] = ds['theta_v_e'][j]

                # And set the plume area to a nan
                ds['a_p'][j] = np.nan
            break
    
    return ds