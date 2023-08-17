def global_plot(data, lon, lat, cmap, pvalues=None, right_title="", left_title="", levels=np.arange(-4, 4.1, .1)):
    # plot map of global data with central longitude 180
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    img = ax1.contourf(
        lon, lat, data,
        transform=ccrs.PlateCarree(), cmap=cmap,
        extend="both",
        levels=levels,
    )
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')

    # East Pacific South America
    bound_region(ax1, lons=(-95, -70), lats=(-40, -5), color="black")
    # Southern Ocean
    bound_region(ax1, lons=(-180, -75), lats=(-70, -50), color="black")
    # East Pacific
    bound_region(ax1, lons=(-135, -80), lats=(-5, 5), color="black")
    # West Pacific
    bound_region(ax1, lons=(110, 165), lats=(-5, 5), color="black")
    
    # Draw triangular region
    lat_min, lat_max = -40, 0
    lon_min, lon_max = -180, -70
    ax1.plot(
        [lon_min, lon_max, lon_max, lon_min], 
        [lat_max, lat_max, lat_min, lat_max], 
        color="black", linewidth=2, transform=ccrs.PlateCarree(), zorder=10
    )

    

    # Add Stippling
    if pvalues is not None:
        ax1.contourf(
            lon, lat, pvalues,
            colors='none',
            levels=[0, .05, 1],
            hatches=['...', None,],
            transform=ccrs.PlateCarree(), 
        )
        
    ax1.coastlines()
    ax1.set_global()
    ax1.set_title(left_title, loc="left", fontweight="bold")
    ax1.set_title(right_title, loc="right")
    lon_min, lon_max = -180, 180
    lat_min, lat_max = -45, 45
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    # Add a horizontal colorbar
    cbar = plt.colorbar(img, orientation='horizontal')

def bound_region(ax, lons, lats, color):
    lon_min, lon_max = lons
    lat_min, lat_max = lats

    # Draw a black bounding box around region
    ax.plot(
        [lon_min, lon_min, lon_max, lon_max, lon_min], 
        [lat_min, lat_max, lat_max, lat_min, lat_min], 
        color=color, linewidth=2, transform=ccrs.PlateCarree(), zorder=10
    )

TOS_T = xr.open_dataarray(f"data/piControl/rolling_gradient_TOS_CMIP6.nc")
tos = TOS_T.sel(model="CanESM5-1").isel(time=1)

def fix_coords(data):
    data = data.bounds.add_bounds("X")
    data = data.bounds.add_bounds("Y")
    data = xc.swap_lon_axis(data, to=(-180, 180))
    return data

tos = fix_coords(tos.drop(["model", "time"]).rename("TOS").to_dataset())
tos = tos.load()
tos
global_plot(
    data=tos,
    lat=tos.lat,
    lon=tos.lon,
    pvalues=None, 
    levels=np.arange(-1, 1.1, .1),
    cmap="RdBu_r",
    left_title="A", 
    right_title="CMIP6 SST $(K/dec)$"
)
i# Calculate (lat,lon) coordinates of the diagonal of a box
latmin, latmax = -38.75, -1.25
lonmin, lonmax = -178.75, -71.25

RES = 2.5
DY = latmax - latmin
DX = lonmax - lonmin 
dx = RES*round(DX/DY)
dy = RES

print(f"For each latitude step of {dy} degrees, longitude step is {dx}")
latcoords = np.arange(latmax, latmin-dy, -dy)
loncoords = np.arange(lonmin, lonmax+dx, dx)
lonraw = np.arange(lonmin, lonmax+dx, RES)


print(lonraw)
print(latcoords)
print(loncoords)

ctos = tos.sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax))

for i, clon in enumerate(lonraw):
    j = np.where(clon == loncoords)[0]
    nlats = len(latcoords) - j # nlats below diag
    ctos[i, :nlats] = np.full((nlats), np.nan) 
    
print(ctos)

latcoords = np.arange(latmax, latmin-dy, -dy)
loncoords = np.arange(lonmin, lonmax+dx, dx)
lonraw = np.arange(lonmin, lonmax+dx, RES)


print(lonraw)
print(latcoords)
print(loncoords)

ctos = tos["TOS"].sel(lon=slice(lonmin, lonmax), lat=slice(latmin, latmax))
print(ctos)

for i, clon in enumerate(lonraw):
    j = np.where(clon == loncoords)[0]

    if i == ctos.shape[1]: break

    print("j prior: ", j)
    if len(j) == 0: 
        j = jold
    else: 
        j = j[0]
        
    print("j: ", j)
    nlats = int(len(latcoords) - j) # nlats below diag
    print("nlats: ", nlats)
    ctos[:nlats, i] = np.full((nlats), np.nan) 
    
    jold = j
    
print(ctos)
