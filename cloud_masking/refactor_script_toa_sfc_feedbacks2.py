# %% [markdown]
# ## Compute kernel-derived TOA and SFC feedbacks using Huang and Huang kernels
# This is a modified version of the notebook **Zelinka_TOA _SFC_feedbacks2.ipynb** 
# 
# Here we are calculating feedbacks using monthly-resolved annual cycle and monthly resolved anomalies from first 150 years of a piControl simulation. The result is not normalized by global mean temperature (i.e. result is now in units Wm^-2 instead of Wm^-2K^-1).
# 
# Modifications:
# 
# - Previously: DATA[var] contained two datasets DATA[var][0] = (time: 12, lat, lon, plev) representing climatology in first 20 years of 4xC02 and DATA[var][1] representing climatology in last 20 years of 4xC02.
# 
# - Previously: DELTA[var] = (DATA[var][1] - DATA[var][0])/GLOBAL_MEAN_TEMP
# 
# - Now, DATA[var] contains two datasets DATA[var][0] = (time: 150*12, lat, lon, plev) representing monthly resolved raw data and DATA[var][1] representing the 12 month climatology tiled 150 times. Tiling makes the time dimension 150*12 so that DELTA (anomalies) can still be calculated as 
# DELTA[var] = DATA[var][1] - DATA[var][0]
# 
# - Note, we also had to tile TOA_KERN and SFC_KERN so that time dimensions are 150*12 and they can be used in calculations with DATA and DELTA. Broadcasting in xarray is not smart enough to figure this out itself.
# 
# - Only other modifications included comments, warning suppression, and a bit of logging. 

# %%
import xsearch as xs
import xcdat as xc
import xarray as xr
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
from typing import List
import gc 
import pickle
import os
from dask.diagnostics import ProgressBar
import traceback

# %%
# import warnings
# warnings.simplefilter("ignore") # Bad Practice but lets suppress xarray warnings for now. 

plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 12
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.handletextpad'] = 0.4

# %%
import json
with open('/home/zelinka1/scripts/coast.json') as json_file:
    data = json.load(json_file)
    coastlat = data['lat']
    coastlon = data['lon360']

# %% [markdown]
# ## Define useful functions

# %%
def add_to_dataset(DATA,name):
    # DS = xr.Dataset({name:(('time','lat','lon'),DATA.data)},
    # coords={'time': TIME,'lat': LAT,'lon': LON}) 
    DS = xr.Dataset({name:(('time','lat','lon'), DATA.data)}) # ZIE (drop coords)
    DS.lat.attrs['axis'] = 'Y'
    DS.lon.attrs['axis'] = 'X'
    DS2 = DS.bounds.add_missing_bounds()
    return DS2[name]

def xarray_time_to_monthly(ds):
    """
    Converts xarray from dims (time of type np.datatype64[M]) to (year, month) where year are integers and month are integers from 1 to 12

    Arguments:
    -----------
        ds [Dataset, DataArray](..., time)

    Returns:
    --------
        ds [Dataset, DataArray](..., year, month)
    """
    year = ds.time.dt.year
    month = ds.time.dt.month

    # assign new coords
    ds = ds.assign_coords(year=("time", year.data), month=("time", month.data))

    # reshape the array to (..., "month", "year")
    return ds.set_index(time=("year", "month")).unstack("time")


def xarray_monthly_to_time(df, time_coord):
    """
    Converts xarray from dims (year, month) where year are integers and month are integers from 1 to 12 to (time of type np.datatype64[M])

    Arguments:
    -----------
        ds [Dataset, DataArray](..., year, month)

    Returns:
    --------
        ds [Dataset, DataArray](..., time)
    """
    # get first and last year
    # firstYr, lastYr = df.year.values[0], df.year.values[-1] + 1
    
    # create time dimension
    df = df.stack(time=["year", "month"])
    
    # set time dimensions using first and last yr
    # df["time"] = np.arange(f"{firstYr}-01", f"{lastYr}-01", dtype="datetime64[M]")
    df["time"] = time_coord
    
    return df

def multiply_monthly_w_time(kern, delta):
    """multiply kern (time 1 - 12 ) with delta (time 1 - 1800)

    Args:
        kern (_type_): _description_
        delta (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    prod = xarray_time_to_monthly(delta)*kern
    prod = xarray_monthly_to_time(prod, time_coord=delta.time)

    return prod



# %%
def find_nearest(array, value):
#     Find the index of the item in the array nearest to the value
#     Args:
#         array: array
#         value: value be found
#     Returns:
#         int: index of value in array
#         https://tropd.github.io/pytropd/helper.html#functions.find_nearest

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# %%
def TropD_Calculate_TropopauseHeight(t, p, Z=None):
#     Calculate the Tropopause Height in isobaric coordinates 
#         Based on the method described in Birner (2010), according to the WMO definition: first level at which the lapse rate <= 2K/km and for which the lapse rate <= 2K/km in all levels at least 2km above the found level 
#         Args:
#             T: Temperature array of dimensions (latitude, levels) on (longitude, latitude, levels)
#             P: pressure levels in hPa
#             Z (optional): geopotential height [m] or any field with the same dimensions as T
#         Returns:
#         ndarray or tuple: 
#           If Z = None, returns Pt(lat) or Pt(lon,lat), the tropopause level in hPa 
#           If Z is given, returns Pt and Ht with shape (lat) or (lon,lat). The field Z evaluated at the tropopause. For Z=geopotential height, Ht is the tropopause altitude in m 

#     Taken from https://tropd.github.io/pytropd/_modules/functions.html#TropD_Calculate_TropopauseHeight
#     Modified by M. Zelinka

    T = np.array(t)
    P = np.array(p)
    
    import scipy as sp
    Rd = 287.04
    Cpd = 1005.7
    g = 9.80616
    k = Rd/Cpd
    PI = (np.linspace(1000,1,1000)*100)**k
    Factor = g/Cpd * 1000

    if len(np.shape(T)) == 2:
        T = np.expand_dims(T, axis=0)
        Z = np.expand_dims(Z, axis=0)
    # make P monotonically decreasing
    if P[-1] > P[0]:
        P = np.flip(P,0)
        T = np.flip(T,2)
        if Z.any():
            Z = np.flip(Z,2)

    Pk = np.tile((P*100)**k, (np.shape(T)[0], np.shape(T)[1], 1))
    Pk2 = (Pk[:,:,:-1] + Pk[:,:,1:])/2

    T2 = (T[:,:,:-1] + T[:,:,1:])/2
    Pk1 = np.squeeze(Pk2[0,0,:])

    Gamma = (T[:,:,1:] - T[:,:,:-1])/(Pk[:,:,1:] - Pk[:,:,:-1]) * Pk2 / T2 * Factor
    Gamma = np.reshape(Gamma, (np.shape(Gamma)[0]*np.shape(Gamma)[1], np.shape(Gamma)[2]))

    T2 = np.reshape(T2, (np.shape(Gamma)[0], np.shape(Gamma)[1]))
    Pt = np.zeros((np.shape(T)[0]*np.shape(T)[1], 1))

    for j in range(np.shape(Gamma)[0]):
        G_f = sp.interpolate.interp1d(Pk1, Gamma[j,:], kind='linear', fill_value='extrapolate')
        G1 = G_f(PI)
        T2_f = sp.interpolate.interp1d(Pk1,T2[j,:], kind='linear', fill_value='extrapolate')
        T1 = T2_f(PI)
        idx = np.squeeze(np.where((G1 <=2) & (PI < (550*100)**k) & (PI > (75*100)**k)))
        Pidx = PI[idx] 

    #     if np.size(Pidx): # MDZ edit
        if np.size(Pidx)>1: # MDZ edit
            for c in range(len(Pidx)):
                dpk_2km =  -2000 * k * g / Rd / T1[c] * Pidx[c]
                idx2 = find_nearest(Pidx[c:], Pidx[c] + dpk_2km)

                if sum(G1[idx[c]:idx[c]+idx2+1] <= 2)-1 == idx2:
                    Pt[j]=Pidx[c]
                    break
        elif np.size(Pidx)==1: # MDZ edit
            Pt[j]=Pidx # MDZ edit
            break # MDZ edit
        else:
            Pt[j] = np.nan



    Pt = Pt ** (1 / k) / 100

    if Z:
        Zt =  np.reshape(Z, (np.shape(Z)[0]*np.shape(Z)[1], np.shape(Z)[2]))
        Ht =  np.zeros((np.shape(T)[0]*np.shape(T)[1]))

        for j in range(np.shape(Ht)[0]):
            f = sp.interpolate.interp1d(P, Zt[j,:])
            Ht[j] = f(Pt[j])

        Ht = np.reshape(Ht, (np.shape(T)[0], np.shape(T)[1]))
        Pt = np.reshape(Pt, (np.shape(T)[0], np.shape(T)[1]))
        return Pt, Ht

    else:
        Pt = np.reshape(Pt, (np.shape(T)[0], np.shape(T)[1]))
        return Pt

# %%
def get_troposphere(pwts,T1):
    pwts2 = np.transpose(np.tile(pwts,(len(lon),len(lat),1)),[2,0,1])
    TROP = xr.DataArray(np.zeros(T1.shape), # ZIE edit (change TIME -> T1.time)
        coords={'time': T1.time,'lev': PLEV.values,'lat': LAT,'lon': LON})
    FULL = xr.DataArray(np.zeros(T1.shape),
        coords={'time': T1.time,'lev': PLEV.values,'lat': LAT,'lon': LON})
    # T1 = T1.load() # Loading data into memory speeds up calculation
    for t in tqdm(range(T1.time.size)): # ZIE edit (12 -> T1.time.size = 1800 and add tqdm) This may take a lot longer
        ZT = T1.isel(time=t).mean('lon').transpose("lat","lev")
        P = ZT.lev/100
        pt = TropD_Calculate_TropopauseHeight(ZT, P)
        tropp = 100*np.tile(pt,(len(PLEV),len(lon),1))
        Tlev = np.transpose(np.tile(ZT.lev,(len(lon),len(lat),1)),[2,0,1])
        TROP[t,:] = np.transpose(np.where(Tlev < tropp, 0, pwts2),[0,2,1])
        FULL[t,:] = np.transpose(pwts2,[0,2,1])

    # plt.subplots()
    # plt.pcolor(lat,PLEV,np.nanmean(np.nanmean(newdT,0),-1))
    # plt.colorbar()
    # plt.subplots()
    # plt.plot(T1,PI**(1/k))
    return(TROP,FULL)

# %%
def calc_norm_dq(DATA_RAW, DATA_CLIM,dTAS):
    # Compute the normalized change in hus using simplified version of what I did in the past:
    q0 = DATA_CLIM['hus']
    q1 = DATA_RAW['hus']

    numer=(np.log(q1)-np.log(q0))/dTAS
    T0 = DATA_CLIM['ta']-273
    T1 = T0+1
    # Bolton (1980) sat vapor pressure formula
    sat0 = 6.112*np.exp((17.67*T0)/(T0 + 243.5))
    sat1 = 6.112*np.exp((17.67*T1)/(T1 + 243.5))
    # Convert from vapor pressure [hPa] to mixing ratio [kg/kg]:
    p = sat0.lev/100
    r0 = 0.622*(sat0/(p-sat0))
    r1 = 0.622*(sat1/(p-sat1))
    # Convert from mixing ratio [kg/kg] to specific humidity [kg/kg]:
    qs0 = r0/(1+r0)
    qs1 = r1/(1+r1)
    denom = (np.log(qs1)-np.log(qs0))
    norm_dq = numer/denom
    return norm_dq

# %%
def compute_fbks(KERN,DELTA, DATA_RAW, DATA_CLIM, dTAS, flag):

    FBK=xr.Dataset()
    
    norm_dq = calc_norm_dq(DATA_RAW,DATA_CLIM, dTAS)
    
    #---------------------------------------------
    # Get pressure weights for the full atmosphere and for the troposphere only
    #---------------------------------------------
    TROP,FULL = get_troposphere(pwts,DATA_RAW['ta']) # ZIE Edit (piControl monthly resolved values)
    print("Finished Tropopause Calculation")
    
    #---------------------------------------------
    # ALL-SKY FEEDBACKS:
    #---------------------------------------------
    FBK['alb'] = multiply_monthly_w_time(KERN['alb_all'], DELTA['alb'])

    prod = multiply_monthly_w_time(KERN['wv_lw_all'], norm_dq)   
    FBK['wv_lw'] = (prod*TROP).sum('lev') # sum vertically w/wts
    prod = multiply_monthly_w_time(KERN['wv_sw_all'], norm_dq)
    FBK['wv_sw'] = (prod*TROP).sum('lev') # sum vertically w/wts
#     FBK['wv_net'] = FBK['wv_lw'] + FBK['wv_sw'] 
    
    sfcPL_all = multiply_monthly_w_time(KERN['ts_all'], DELTA['ts'])
    prod1 = (multiply_monthly_w_time(KERN['ta_all'], DELTA['ta'])).transpose('lev','time','lat','lon')
    prod2 = multiply_monthly_w_time(KERN['ta_all'].transpose('lev','month','lat','lon'), DELTA['ts'])
#     # Mask prod1 and prod2 in same places:
    prod3 = xr.where(prod1==0,0,prod2)
    ta_all = (prod1*TROP).sum('lev') # sum vertically w/wts
    atmPL_all = (prod3*TROP).sum('lev') # sum vertically w/wts
# #     FBK['PL_up'] = sfcPL_all
# #     FBK['PL_dn'] = atmPL_all
    FBK['PL'] = sfcPL_all + atmPL_all
    FBK['LR'] = ta_all - atmPL_all


    # HELD AND SHELL FEEDBACKS:
    # (original water vapor kernel) x (atmospheric temperature response):
    prod = multiply_monthly_w_time(KERN['wv_lw_all'], DELTA['ta'])
    HS_wv_lw_all = (prod*TROP).sum('lev') # sum vertically w/wts
    prod = multiply_monthly_w_time(KERN['wv_sw_all'],DELTA['ta'])
    HS_wv_sw_all = (prod*TROP).sum('lev') # sum vertically w/wts

    # Compute \\Fixed RH\\ T kernel following Held and Shell 2012
    NET_Tkern_fxRH = KERN['ta_all'] + KERN['wv_lw_all'] + KERN['wv_sw_all']
    prod = multiply_monthly_w_time(NET_Tkern_fxRH,DELTA['ta'])
    fxRH_ta_all = (prod*TROP).sum('lev') # sum vertically w/wts
    prod = multiply_monthly_w_time(NET_Tkern_fxRH.transpose('lev','month','lat','lon'), DELTA['ts'])
    fx_RH_atmPL_all = (prod*TROP).sum('lev') # sum vertically w/wts
    FBK['HS_PL'] = sfcPL_all + fx_RH_atmPL_all
    FBK['HS_LR'] = fxRH_ta_all - fx_RH_atmPL_all
    # RH feedback  = traditional WV fbk minus (orig. WVkern x Ta response)
    FBK['HS_RH'] = (FBK['wv_sw'] - HS_wv_sw_all) + (FBK['wv_lw'] - HS_wv_lw_all)

    #---------------------------------------------
    # CLEAR-SKY FEEDBACKS:
    #---------------------------------------------
    FBK['alb_clr'] = multiply_monthly_w_time(KERN['alb_clr'], DELTA['alb'])
    
    prod = multiply_monthly_w_time(KERN['wv_lw_clr'],norm_dq)
    FBK['wv_lw_clr'] = (prod*TROP).sum('lev') # sum vertically w/wts
    prod = multiply_monthly_w_time(KERN['wv_sw_clr'],norm_dq)
    FBK['wv_sw_clr'] = (prod*TROP).sum('lev') # sum vertically w/wts
    # FBK['wv_net_clr'] = FBK['wv_lw_clr'] + FBK['wv_sw_clr']     
    
    sfcPL_clr = multiply_monthly_w_time(KERN['ts_clr'], DELTA['ts'])
    prod1 = (multiply_monthly_w_time(KERN['ta_clr'], DELTA['ta'])).transpose('lev','time','lat','lon')
    prod2 = multiply_monthly_w_time(KERN['ta_clr'].transpose('lev','month','lat','lon'), DELTA['ts'])
    # # Mask prod1 and prod2 in same places:
    prod3 = xr.where(prod1==0,0,prod2)
    ta_clr = (prod1*TROP).sum('lev') # sum vertically w/wts
    atmPL_clr = (prod3*TROP).sum('lev') # sum vertically w/wts
    FBK['PL_clr'] = sfcPL_clr + atmPL_clr
    FBK['LR_clr'] = ta_clr - atmPL_clr

    # HELD AND SHELL FEEDBACKS:
    # (original water vapor kernel) x (atmospheric temperature response):
    # prod = KERN['wv_lw_clr']*DELTA['ta']
    # HS_wv_lw_clr = (prod*TROP).sum('lev') # sum vertically w/wts
    # prod = KERN['wv_sw_clr']*DELTA['ta']
    # HS_wv_sw_clr = (prod*TROP).sum('lev') # sum vertically w/wts

    # Compute \\Fixed RH\\ T kernel following Held and Shell 2012
    # NET_Tkern_fxRH = KERN['ta_clr'] + FBK['wv_lw_clr'] + FBK['wv_sw_clr']
    # prod = NET_Tkern_fxRH*DELTA['ta']
    # fxRH_ta_clr = (prod*TROP).sum('lev') # sum vertically w/wts
    # prod = NET_Tkern_fxRH.transpose('lev','time','lat','lon')*DELTA['ts']
    # fx_RH_atmPL_clr = (prod*TROP).sum('lev') # sum vertically w/wts
    # FBK['HS_PL_clr'] = sfcPL_clr + fx_RH_atmPL_clr
    # FBK['HS_LR_clr'] = fxRH_ta_clr - fx_RH_atmPL_clr
    # RH feedback  = traditional WV fbk minus (orig. WVkern x Ta response)
    # FBK['HS_RH_clr'] = (FBK['wv_sw_clr'] - HS_wv_sw_clr) + (FBK['wv_lw_clr'] - HS_wv_lw_clr)

    #---------------------------------------------
    # CLOUD FEEDBACKS:
    #---------------------------------------------
    # Cloud masking terms (these should be troposphere only)
    LW_Planck_mask = FBK['PL_clr'] - FBK['PL']
    LW_LR_mask = FBK['LR_clr'] - FBK['LR']
    LW_WV_mask = FBK['wv_lw_clr'] - FBK['wv_lw']
    SW_alb_mask = FBK['alb_clr'] - FBK['alb']
    SW_WV_mask = FBK['wv_sw_clr'] - FBK['wv_sw']

    # total corrections necessary to adjust the delta CRE into cloud feedback:
    LW_sum_correct = LW_Planck_mask + LW_LR_mask + LW_WV_mask
    SW_sum_correct = SW_WV_mask + SW_alb_mask
    NET_sum_correct = LW_sum_correct + SW_sum_correct
    
    if flag=='toa':
        FBK['cld_lw'] = DELTA['LWCRE'] + LW_sum_correct
        FBK['cld_sw'] = DELTA['SWCRE'] + SW_sum_correct
        FBK['cld_net'] = DELTA['netCRE'] + NET_sum_correct
        FBK['LWtrue'] = -DELTA['rlut']
        FBK['SWtrue'] = DELTA['rsdt'] - DELTA['rsut']
        FBK['NETtrue'] = DELTA['rndt']
        FBK['LWtrue_clr'] = -DELTA['rlutcs']
        FBK['SWtrue_clr'] = DELTA['rsdt'] - DELTA['rsutcs']
        FBK['NETtrue_clr'] = DELTA['rndtcs'] 
    else:
        FBK['cld_lw'] = DELTA['sfc_LWCRE'] + LW_sum_correct
        FBK['cld_sw'] = DELTA['sfc_SWCRE'] + SW_sum_correct
        FBK['cld_net'] = DELTA['sfc_netCRE'] + NET_sum_correct
        # FBK['LWtrue'] = DELTA['rlds'] - DELTA['rlus']
        # FBK['SWtrue'] = DELTA['rsds'] - DELTA['rsus']
        FBK['NETtrue'] = DELTA['rnds']
        # FBK['LWtrue_clr'] = DELTA['rldscs'] - DELTA['rlus']
        # FBK['SWtrue_clr'] = DELTA['rsdscs'] - DELTA['rsuscs']
        # FBK['NETtrue_clr'] = DELTA['rndscs']
   
    #---------------------------------------------
    # SUMMATION OF FEEDBACKS:
    #---------------------------------------------
    FBK['sum_lw'] = FBK['PL'] + FBK['LR'] + FBK['wv_lw'] + FBK['cld_lw']
    FBK['sum_sw'] = FBK['wv_sw'] + FBK['alb'] + FBK['cld_sw']
    FBK['sum_net'] = FBK['sum_lw'] + FBK['sum_sw']
    # FBK['resid_lw'] = FBK['LWtrue'] - FBK['sum_lw']
    # FBK['resid_sw'] = FBK['SWtrue'] - FBK['sum_sw']
    FBK['resid_net'] = FBK['NETtrue'] - FBK['sum_net']
    
    # FBK['sum_lw_clr'] = FBK['PL_clr'] + FBK['LR_clr'] + FBK['wv_lw_clr']
    # FBK['sum_sw_clr'] = FBK['wv_sw_clr'] + FBK['alb_clr']
    # FBK['sum_net_clr'] = FBK['sum_lw_clr'] + FBK['sum_sw_clr']
    # FBK['resid_lw_clr'] = FBK['LWtrue_clr'] - FBK['sum_lw_clr']
    # FBK['resid_sw_clr'] = FBK['SWtrue_clr'] - FBK['sum_sw_clr']
    # FBK['resid_net_clr'] = FBK['NETtrue_clr'] - FBK['sum_net_clr']

    FBK.lat.attrs['axis'] = 'Y'
    FBK.lon.attrs['axis'] = 'X'
    FBK = FBK.bounds.add_missing_bounds()    
    # FBK['lat_bnds'] = Ybnds
    # FBK['lon_bnds'] = Xbnds
    
    
    return(FBK)
    


# %%
def clr_sky_linearity(FBK,flag):
    # CLEAR-SKY LINEARITY TEST   

    plt.subplots()   
    var='LWtrue_clr'
    GL = np.round(FBK.spatial.average(var)[var].mean('time').values,2)
    plt.plot(lat,FBK[var].mean(('time','lon')),label=var+' ['+str(GL)+']',color='C1',ls='--')
    var='SWtrue_clr'
    GL = np.round(FBK.spatial.average(var)[var].mean('time').values,2)
    plt.plot(lat,FBK[var].mean(('time','lon')),label=var+' ['+str(GL)+']',color='C2',ls='--')
    var='NETtrue_clr'
    GL = np.round(FBK.spatial.average(var)[var].mean('time').values,2)
    plt.plot(lat,FBK[var].mean(('time','lon')),label=var+' ['+str(GL)+']',color='C0',ls='--')

    var='sum_lw_clr'
    GL = np.round(FBK.spatial.average(var)[var].mean('time').values,2)
    plt.plot(lat,FBK[var].mean(('time','lon')),label=var+' ['+str(GL)+']',color='C1',ls='-')
    var='sum_sw_clr'
    GL = np.round(FBK.spatial.average(var)[var].mean('time').values,2)
    plt.plot(lat,FBK[var].mean(('time','lon')),label=var+' ['+str(GL)+']',color='C2',ls='-')
    var='sum_net_clr'
    GL = np.round(FBK.spatial.average(var)[var].mean('time').values,2)
    plt.plot(lat,FBK[var].mean(('time','lon')),label=var+' ['+str(GL)+']',color='C0',ls='-')

    plt.legend(ncol=2)
    plt.axhline(y=0,color='k')
    plt.title(modripf+' ['+flag+']',loc='left')
    # plt.savefig('/home/zelinka1/figures/'+era+'/feedbacks/'+exp+'/clrsky_linearity_'+modripf+'_'+flag+'.png')
    
    


# %% [markdown]
# ## Read in Huang and Huang kernels and place into a dataset

# %%
# Huang kernel grid:
kern=xc.open_dataset('/home/zelinka1/kernels/huang/era5/nodp/ERA5_kernel_ta_nodp_SFC.nc',decode_times=False)
kern = kern.sortby('latitude', ascending=True)  # -90 to 90
lat = np.array(kern.latitude)
lon = np.array(kern.longitude)
lat_a = xc.create_axis("lat", lat)
lon_a = xc.create_axis("lon", lon)
output_grid = xc.regridder.grid.create_grid(y=lat_a, x=lon_a)
                  
K={}
gg = glob.glob('/home/zelinka1/kernels/huang/era5/nodp/ERA5_kernel_*.nc')
gg.sort()
for g in gg:
    kern=xc.open_dataset(g,decode_times=False)
    kern = kern.sortby('latitude', ascending=True)  # -90 to 90
    this = kern.rename({'month': 'time','latitude': 'lat','longitude': 'lon'})
    for var in kern.data_vars:
        if 'bnds' in var:
            continue
        plt.subplots()
        this[var].mean(('time','lon')).plot()
        name = g.split('/')[-1].split('kernel_')[-1].split('.')[0]+'_'+var.split('_')[-1]
        plt.title(name)
        
        # Regrid to kernel grid:
        output = this.regridder.horizontal(var, output_grid, tool='xesmf', method='bilinear')

        K[name] = output#kern[var]
        
# Convert these back into an xarray dataset:
TIME = K['ta_nodp_TOA_all'].time
PLEV = 100*K['ta_nodp_TOA_all'].level
LAT = K['ta_nodp_TOA_all'].lat
LON = K['ta_nodp_TOA_all'].lon

# # TOA KERNEL:
# DS = xr.Dataset(
# {
#     'wv_lw_all':(('time','lev','lat','lon'),K['wv_lw_nodp_TOA_all']['TOA_all'].data),
#     'wv_lw_clr':(('time','lev','lat','lon'),K['wv_lw_nodp_TOA_clr']['TOA_clr'].data),   
#     'wv_sw_all':(('time','lev','lat','lon'),K['wv_sw_nodp_TOA_all']['TOA_all'].data),
#     'wv_sw_clr':(('time','lev','lat','lon'),K['wv_sw_nodp_TOA_clr']['TOA_clr'].data),
   
#     'ta_all':(('time','lev','lat','lon'),K['ta_nodp_TOA_all']['TOA_all'].data),
#     'ta_clr':(('time','lev','lat','lon'),K['ta_nodp_TOA_clr']['TOA_clr'].data),
#     'alb_all':(('time','lat','lon'),K['alb_TOA_all']['TOA_all'].data),
#     'alb_clr':(('time','lat','lon'),K['alb_TOA_clr']['TOA_clr'].data),
#     'ts_all':(('time','lat','lon'),K['ts_TOA_all']['TOA_all'].data),    
#     'ts_clr':(('time','lat','lon'),K['ts_TOA_clr']['TOA_clr'].data),
# },
# coords={'time': TIME,'lev': PLEV.values,'lat': LAT,'lon': LON},
# ) 
# DS.lat.attrs['axis'] = 'Y'
# DS.lon.attrs['axis'] = 'X'
# TOA_KERN = DS.bounds.add_missing_bounds()

# SFC KERNEL:
DS = xr.Dataset(
{
    'wv_lw_all':(('time','lev','lat','lon'),K['wv_lw_nodp_SFC_all']['SFC_all'].data),
    'wv_lw_clr':(('time','lev','lat','lon'),K['wv_lw_nodp_SFC_clr']['SFC_clr'].data),
    'wv_sw_all':(('time','lev','lat','lon'),K['wv_sw_nodp_SFC_all']['SFC_all'].data),
    'wv_sw_clr':(('time','lev','lat','lon'),K['wv_sw_nodp_SFC_clr']['SFC_clr'].data),
    'ta_all':(('time','lev','lat','lon'),K['ta_nodp_SFC_all']['SFC_all'].data),
    'ta_clr':(('time','lev','lat','lon'),K['ta_nodp_SFC_clr']['SFC_clr'].data),   
    'alb_all':(('time','lat','lon'),K['alb_SFC_all']['SFC_all'].data),
    'alb_clr':(('time','lat','lon'),K['alb_SFC_clr']['SFC_clr'].data),
    'ts_all':(('time','lat','lon'),K['ts_SFC_all']['SFC_all'].data),
    'ts_clr':(('time','lat','lon'),K['ts_SFC_clr']['SFC_clr'].data),    
},
coords={'time': TIME,'lev': PLEV.values,'lat': LAT,'lon': LON},
) 
DS.lat.attrs['axis'] = 'Y'
DS.lon.attrs['axis'] = 'X'
SFC_KERN = DS.bounds.add_missing_bounds()

# %%
# lay_thick = np.gradient(np.array(TOA_KERN.lev)) # typically 2500 Pa = 25 hPa
# pwts = -lay_thick/100/100 # convert from Pa to hPa, then to per 100 hPa
# pwts,lay_thick,KERN.lev
pwts=np.ones(len(SFC_KERN.lev))


# %%
variables = ['tas','huss','ta', 'hus', 'rlut','rlutcs','rsds','rsdt','rsus','rsut','rsutcs','ts',
             'rsdscs','rsuscs','rlds','rldscs','rlus','pr','hfls','hfss']
# variables = ['tas','huss', 'rsds','rsus', 'ta', 'hus'] # ZIE Edit (For Testing)

eras = ['CMIP5','CMIP6'] # user choice
everything={}
keep={}
for era in eras:
    modripfs=[]
    everything[era]={}
    if era=='CMIP5':
        exps = ['piControl'] # ZIE edit (abrupt-4xCO2 -> piControl)
        ACT = '*'
    else:
        exps = ['piControl'] # ZIE edit (abrupt-4xCO2 -> piControl)
        ACT = '*'
    for exp in exps:
        everything[era][exp]={}
        for var in variables:
            everything[era][exp][var]={}
            pathDict = xs.findPaths(exp, var, 'mon', mip_era=era, cmipTable='*mon', activity=ACT)#, filterRetracted=False)
            models = xs.getGroupValues(pathDict, 'model')
            for mod in models:
                
                everything[era][exp][var][mod]={}
                pathDict = xs.findPaths(exp, var, 'mon', mip_era=era, cmipTable='*mon', activity=ACT, model=mod)#, filterRetracted=False)
                ripfs = xs.getGroupValues(pathDict, 'member')
                for ripf in ripfs:
                    pathDict = xs.findPaths(exp, var, 'mon', mip_era=era, cmipTable='*mon',activity=ACT, model=mod, member=ripf)#, filterRetracted=False)
                    dpath=list(pathDict.keys())
                    if len(dpath)==1:
                        everything[era][exp][var][mod+'.'+ripf] = dpath[0]
                        modripfs.append(mod+'.'+ripf)
                    else:
                        break # need to investigate why more than one path survived...

    # Need to find which models.ripfs have all variables for both experiments:
    keep[era]=[]
    modripfs.sort()
    for modripf in np.unique(modripfs):
#         if 'r1i' not in modripf:
#             continue
        names=[]
        for var in variables:
            for exp in exps:
                try:
                    names.append((everything[era][exp][var][modripf]))
                except:
                    print('No '+var+' data for '+modripf+'.'+exp)
                    continue
        if len(names) == len(variables) * len(exps):
            keep[era].append(modripf)


# %%
# Huang kernel grid:
kern=xc.open_dataset('/home/zelinka1/kernels/huang/era5/nodp/ERA5_kernel_ta_nodp_SFC.nc', decode_times=False)
kern = kern.sortby('latitude', ascending=True)  # -90 to 90
kern
eras = ['CMIP5', 'CMIP6']
z = xc.create_axis('lev', np.array(PLEV))
want = xc.create_grid(z=z)
want


def get_data(models: List[str], fpath: str):
    """
    Creates two datasets for each model: one for raw values and the other for the climatology.
    This refactor requires more disk space than old version but is far less memory intensive
    and easier to debug. 
    After the full pipeline is run the data files generated here can be deleted.
    """
    SKIP_MODELS = ["CAS-ESM2-0.r1i1p1f1", "ICON-ESM-LR.r1i1p1f1", "GISS-E2-1-G.r1i1p1f3", "CESM1-BGC.r1i1p1", "CESM1-FASTCHEM.r1i1p1", "CCSM4.r1i1p1"]

    for i, modripf in enumerate(models):
        DATA_CLIM = xr.Dataset()
        DATA_RAW = xr.Dataset()
        DELTA = xr.Dataset()
        try: 
        # if True:
            # Skip already processed models and models that have known issues
            save_path_clim = fpath+'/DATA_CLIM_'+modripf+'.nc'
            save_path_raw = fpath+'/DATA_RAW_'+modripf+'.nc'
            save_path_delta = fpath+'/DELTA_'+modripf+'.nc'
            if os.path.exists(save_path_clim) and os.path.exists(save_path_raw) and os.path.exists(save_path_delta):
                continue
            if modripf in SKIP_MODELS:
                continue # two files contain year 50

            t0 = time.time() # time each model

            # get model name (mod) and version (ripf)
            mod,ripf = modripf.split('.')

            # Only process r1i1p1 and r1i1p1f1
            if (ripf != 'r1i1p1') and (ripf != 'r1i1p1f1'): continue

            skip=False   
            for var in variables:   
                pathDict = xs.findPaths(exp, var, 'mon', mip_era=era, realm='atmos',cmipTable='Amon', model=mod, member=ripf)
                dpath=list(pathDict.keys())
                filename = dpath[0] 
                if var=='tas':  
                    print('----reading in '+filename)
                ds0 = xc.open_mfdataset(filename+'*nc',decode_times=True)
                print(f"----starting {var}")
                
                # Check to make sure this model has at least 150 years:
                L = len(ds0.time)
                if L<150*12:
                    print('Length of '+modripf+' is only '+str(L/12)+' years')
                    raise ValueError('Model too short') # This will be caught lower down
                    
                Jan = np.where(ds0.time.dt.month[:12]==1)[0][0] # 0 in all models; 1 in HadGEM2-ES
                
                for per in range(2):
                    # Select first 150 years
                    ds = ds0.isel(time=slice(Jan+0, Jan+150*12))
                    time_idx = ds.time

                    if ds.time.dt.month[0]!=1:
                        print('Month 1 is not January')
                        break # skip if first month is not January
                    
                    if per == 0: # Climatology
                        # (index 0 is climatological monthly-resolved annual cycle)
                        ds = ds.temporal.climatology(var, 'month', weighted=True) 
                    else: # (index 1 is monthly raw values) 
                        pass

                    # Regrid to kernel horizontal grid:
                    ds = ds.regridder.horizontal(var, output_grid, tool='xesmf', method='bilinear') #.load()
                    # Regrid to kernel vertical grid:
                    if (var=='ta') or (var=='hus'): 
                        ds = ds.rename({'plev':'lev'}) # rename plev to lev
                        ds.lev.attrs['axis']='Z'
                        ds = ds.chunk({'lev':-1}) # ZIE Rechunk ds2 along lev
                        z = xc.create_axis('lev', np.array(PLEV)) # ZIE (new version of xcdat requires axis obj)
                        want = xc.create_grid(z=z)
                        if 'plev_bnds' in ds:
                            ds = ds.rename({'plev_bnds':'lev_bnds'})
                        ds = ds.regridder.vertical(var, want, method='linear')#, target_data=orig)

                        # Fill lower levels for climatology here before we broadcast.
                        # This avoids redundant calculations.
                        if per == 0:
                            if var == 'ta':
                                surface_filler = DATA_CLIM['tas'].isel(time=slice(0,12)).copy(deep=True)
                            if var == 'hus':
                                surface_filler = DATA_CLIM['huss'].isel(time=slice(0,12)).copy(deep=True)
                            surface_filler['time'] = ds[var].time

                            print(f'Modify Low-Levels {var} clim')
                            ds[var] = ds[var].load()
                            ds[var][:,:15,:] = xr.where(np.isnan(ds[var][:,:15,:]),
                                                            surface_filler,
                                                            ds[var][:,:15,:])
                    
                    if per == 0: # Broadcast Climatology
                        ds = xr.concat([ds]*150, dim='time') # Tile climatology to full time series
                        ds['time'] = time_idx
                        DATA_CLIM[var] = ds[var]
                    else: # Raw Values
                        DATA_RAW[var] = ds[var]

            if modripf == 'IITM-ESM.r1i1p1f1':
                DATA_CLIM['rldscs'] = -DATA_CLIM['rldscs']
                DATA_RAW['rldscs'] = -DATA_RAW['rldscs']         

            ## DOUBLE CHECK LATITUDE/LONGITUDE BOUNDS
            DATA_RAW.lat.attrs['axis'] = 'Y'
            DATA_RAW.lon.attrs['axis'] = 'X'
            DATA_RAW = DATA_RAW.bounds.add_missing_bounds()

            DATA_CLIM.lat.attrs['axis'] = 'Y'
            DATA_CLIM.lon.attrs['axis'] = 'X'
            DATA_CLIM = DATA_CLIM.bounds.add_missing_bounds()
                
            def _compute_additional_fields(data):
                alb = data['rsus']/data['rsds']
                data['alb'] = add_to_dataset(alb,'alb')
                rndt = data['rsdt']- data['rsut'] - data['rlut']
                data['rndt'] = add_to_dataset(rndt,'rndt')
                rndtcs = data['rsdt'] - data['rsutcs'] - data['rlutcs']
                data['rndtcs'] = add_to_dataset(rndtcs,'rndtcs')        
                swcre = data['rsutcs'] - data['rsut']
                data['SWCRE'] = add_to_dataset(swcre,'SWCRE')
                lwcre = data['rlutcs'] - data['rlut']
                data['LWCRE'] = add_to_dataset(lwcre,'LWCRE')
                netcre = data['SWCRE'] + data['LWCRE']
                data['netCRE'] = add_to_dataset(netcre,'netCRE')
                # surface CRE:
                rnds = data['rsds'] + data['rlds'] - data['rsus'] - data['rlus']
                data['rnds'] = add_to_dataset(rnds,'rnds')
                rndscs = data['rsdscs'] + data['rldscs'] - \
                        data['rsuscs'] - data['rlus']
                data['rndscs'] = add_to_dataset(rndscs,'rndscs')
                # sfc_swcre = (down all - up all) - (down clr - up clr) 
                sfc_swcre = (data['rsds'] - data['rsus']) - \
                            (data['rsdscs'] - data['rsuscs'])          
                data['sfc_SWCRE'] = add_to_dataset(sfc_swcre,'sfc_SWCRE')            
                # sfc_lwcre = (down all - up all) - (down clr - up clr) 
                sfc_lwcre = (data['rlds']- data['rlus']) - \
                            (data['rldscs'] - data['rlus'])
                data['sfc_LWCRE'] = add_to_dataset(sfc_lwcre,'sfc_LWCRE')
                sfc_netcre = data['sfc_SWCRE']+ data['sfc_LWCRE']
                data['sfc_netCRE'] = add_to_dataset(sfc_netcre,'sfc_netCRE')
                # Atmopsheric heat flux divergence:
                divF = (data['rsdt'] - data['rsut']) - data['rlut'] - \
                    ((data['rsds'] - data['rsus']) + \
                    (data['rlds'] - data['rlus']) - \
                        data['hfss'] - data['hfls'])
                data['divF'] = add_to_dataset(divF,'divF')

                return data

            print('Compute additional fields')
            DATA_CLIM = _compute_additional_fields(DATA_CLIM)
            DATA_RAW = _compute_additional_fields(DATA_RAW)

            print('Compute DELTAs')
            new = ['alb','rndt','rndtcs','SWCRE','LWCRE','netCRE','rnds','rndscs','sfc_SWCRE','sfc_LWCRE','sfc_netCRE','divF']
            new_vars = variables+new
            dTAS = 1
            Xbnds = DATA_RAW.bounds.get_bounds('X')
            Ybnds = DATA_RAW.bounds.get_bounds('Y')
            for var in new_vars:
                diff = (DATA_RAW[var] - DATA_CLIM[var])/dTAS # ZIE edit
                if var=='alb': # normalize delta albedo by a 1% increase in albedo
                    diff = diff/0.01
                diff.fillna(0)
                DELTA[var] = diff

            DELTA['lat_bnds'] = Ybnds
            DELTA['lon_bnds'] = Xbnds
            DELTA = DELTA.assign_coords(time=DATA_RAW.time)
            
            # Where delta ta is zero, set it equal to delta tas    
            # Where hus/ta are not defined, set them equal to huss/tas
            # (just do this for the lowest 15 levels)
            # This is a bit of a kludge to ensure that the SFC kernels (which are large at low-levels) are multiplied by something nonzero
            print('Modify Low-Levels hus raw')
            DATA_RAW['hus'] = DATA_RAW['hus'].load()
            DATA_RAW['hus'][:,:15,:] = xr.where(np.isnan(DATA_RAW['hus'][:,:15,:]),
                                            DATA_RAW['huss'],
                                            DATA_RAW['hus'][:,:15,:])

            print('Modify Low-Levels ta raw')
            DATA_RAW['ta']= DATA_RAW['ta'].load()
            DATA_RAW['ta'][:,:15,:] = xr.where(np.isnan(DATA_RAW['ta'][:,:15,:]),
                                        DATA_RAW['tas'],
                                        DATA_RAW['ta'][:,:15,:])

            print('Modify Low-Levels delta ta')
            DELTA['ta'] = DELTA['ta'].load()
            DELTA['ta'][:,:15,:] = xr.where(DELTA['ta'][:,:15,:]==0,
                                                DELTA['tas'],
                                                DELTA['ta'][:,:15,:])

            print('Modify Low-Levels delta hus')
            DELTA['hus'] = DELTA['hus'].load()
            DELTA['hus'][:,:15,:] = xr.where(DELTA['hus'][:,:15,:]==0,
                                                DELTA['huss'],
                                                DELTA['hus'][:,:15,:])

            with ProgressBar():
                print("Saving clim data for... "+modripf)
                DATA_CLIM.to_netcdf(save_path_clim)
                print("Saving raw data for... "+modripf)
                DATA_RAW.to_netcdf(save_path_raw)
                print("Saving delta for... "+modripf)
                DELTA.to_netcdf(save_path_delta)

            elapsed = time.time() - t0
            print('I/O for '+modripf+' in '+str(np.round(elapsed/60,2))+' mins')

        except Exception as e:
            print('Error in '+modripf)
            print(e)
            continue


SFC_KERN = SFC_KERN.rename({'time': 'month'})
SFC_KERN["month"] = np.arange(1, 13)

for era in eras:

    if era=='CMIP5':
        exp = 'piControl' # ZIE edit
        fpath = '/home/espinosa10/tropical_pacific_clouds/cloud_masking/cmip5' # ZIE edit
#         continue
    else:
        exp = 'piControl' # ZIE edit
        fpath = '/home/espinosa10/tropical_pacific_clouds/cloud_masking/cmip6' # ZIE edit

    models = sorted(keep[era])
    # get_data(models=models, fpath=fpath)

    # Now compute feedbacks
    for i, modripf in enumerate(models):
        try: 
        # if True:
            if os.path.exists(fpath+'/sfc_fbks_'+modripf+'.nc'):
                print(f"SFC FBKS already exists for {modripf}")
                continue

            print("Starting feedbacks for... "+modripf)

            # get model name (mod) and version (ripf)
            mod,ripf = modripf.split('.')

            # Only process r1i1p1 and r1i1p1f1
            if (ripf != 'r1i1p1') and (ripf != 'r1i1p1f1'): continue
            
            t0 = time.time()
            save_path_clim = fpath+'/DATA_CLIM_'+modripf+'.nc'
            save_path_raw = fpath+'/DATA_RAW_'+modripf+'.nc'
            save_path_delta = fpath+'/DELTA_'+modripf+'.nc'

            if not os.path.exists(save_path_clim): continue
            if modripf == "IPSL-CM6A-LR.r1i1p1f1": continue

            chunks = {'time': 120, 'lat': -1, 'lon': -1, 'lev': -1}
            DATA_RAW = xr.open_dataset(save_path_raw,  chunks=chunks)[['ta', 'hus']]
            DATA_CLIM = xr.open_dataset(save_path_clim, chunks=chunks)[['hus', 'ta']]
            DELTA = xr.open_dataset(save_path_delta, chunks=chunks)
            dTAS = 1

            print('Compute SFC FBKS')
            SFC_FBK = compute_fbks(SFC_KERN, DELTA, DATA_RAW, DATA_CLIM, dTAS, 'sfc')

            with ProgressBar():
                print("Sanity Check: ", SFC_FBK)
                SFC_FBK = SFC_FBK.load()
                print("Sanity Check: ", SFC_FBK)
                SFC_FBK.to_netcdf(fpath+'/sfc_fbks_'+modripf+'.nc')

            elapsed = time.time() - t0
            print('Feedbacks for '+modripf+' in '+str(np.round(elapsed/60,2))+' mins')
        
        except Exception as e:
            print('Error in '+modripf)
            print(traceback.format_exc())
            print(e)
            continue

# if len(SFC_KERN.time) != len(DATA_RAW.time): #     SFC_KERN = xr.concat([SFC_KERN]*150, dim='time') # Broadcast SFC_KERN to full time series
#     SFC_KERN['time'] = DATA_RAW.time
#     SFC_KERN = SFC_KERN.chunk({'time': 120, 'lat': -1, 'lon': -1})

# SFC_KERN['time'] = DATA_RAW.time





