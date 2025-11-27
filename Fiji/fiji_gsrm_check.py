import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fiji vortex center
lon0 = 178.0
lat0 = -17.75

##############################
# 1. LOAD DATA
##############################
gps_cols = ["lon", "lat", "Ve", "Vn", "sigma_E", "sigma_N", "corrEN", "plate"]
gps = pd.read_csv(
    "data/GPS_vectors_after_rotation_PA.dat",
    sep='\\s+',
    comment="#",
    names=gps_cols
)

model_cols = ["lon", "lat", "Ve", "Vn", "_1", "_2", "_3"]
model = pd.read_csv(
    "data/velocity_PA.dat",
    sep='\\s+',
    comment="#",
    names=model_cols
)

print(f"Total GPS stations: {len(gps)}")
print(f"Total model points: {len(model)}")

##############################
# FIX: Normalize longitude to 0-360
##############################
def normalize_lon360(lon):
    """Convert longitude to 0-360 range"""
    return np.where(lon < 0, lon + 360, lon)

gps['lon360'] = normalize_lon360(gps.lon)
model['lon360'] = normalize_lon360(model.lon)
lon0_360 = normalize_lon360(lon0)

print(f"\nLongitude ranges after normalization:")
print(f"GPS: {gps.lon360.min():.1f} to {gps.lon360.max():.1f}")
print(f"Model: {model.lon360.min():.1f} to {model.lon360.max():.1f}")
print(f"Fiji center: {lon0_360:.1f}°E")

##############################
# 2. COORDINATE CONVERSION (Fiji at origin)
##############################
def lonlat_to_local(lon, lat, lon0, lat0):
    """Convert to local Cartesian coordinates with Fiji at (0,0)"""
    Re = 6371e3  # meters
    dlon = np.radians(lon - lon0)
    dlat = np.radians(lat - lat0)
    x = Re * dlon * np.cos(np.radians(lat0))
    y = Re * dlat
    return x, y

# Convert using lon360
gps["x"], gps["y"] = lonlat_to_local(gps.lon360, gps.lat, lon0_360, lat0)
model["x"], model["y"] = lonlat_to_local(model.lon360, model.lat, lon0_360, lat0)

print(f"\nCoordinate system check:")
print(f"Fiji should be at (0, 0)")
print(f"Min x: {gps.x.min()/1e6:.2f} Mm, Max x: {gps.x.max()/1e6:.2f} Mm")
print(f"Min y: {gps.y.min()/1e6:.2f} Mm, Max y: {gps.y.max()/1e6:.2f} Mm")

##############################
# 3. COMPUTE VELOCITY COMPONENTS
##############################
def compute_components(df):
    """Compute radial and tangential components relative to Fiji"""
    r = np.sqrt(df.x**2 + df.y**2)
    
    # Avoid division by zero at origin
    r = np.where(r == 0, 1e-10, r)
    
    ux = df.x / r  # radial unit vector (x component)
    uy = df.y / r  # radial unit vector (y component)
    
    # Radial component (positive = away from Fiji)
    df["Vr"] = df.Ve * ux + df.Vn * uy
    
    # Tangential component (positive = counterclockwise)
    # Perpendicular to radial: rotate radial by 90° counterclockwise
    df["Vt"] = df.Ve * (-uy) + df.Vn * ux
    
    df["r_km"] = r / 1000.0
    df["speed"] = np.sqrt(df.Ve**2 + df.Vn**2)
    
    return df

gps = compute_components(gps)
model = compute_components(model)

##############################
# 4. FILTER TO BROAD REGION (using lon360)
##############################
lat_mask = (gps.lat >= -30.0) & (gps.lat <= -10.0)
lon_mask = (gps.lon360 >= 160.0) & (gps.lon360 <= 220.0)
gps_region = gps[lat_mask & lon_mask].copy()

lat_mask_m = (model.lat >= -30.0) & (model.lat <= -10.0)
lon_mask_m = (model.lon360 >= 160.0) & (model.lon360 <= 220.0)
model_region = model[lat_mask_m & lon_mask_m].copy()

print(f"\nGPS stations in SW Pacific region: {len(gps_region)}")
print(f"Model points in SW Pacific region: {len(model_region)}")

##############################
# 5. CLASSIFY BY SUB-REGION (using lon360)
##############################
def classify_region(lon360, lat):
    """Classify points into Vanuatu, Fiji, or Tonga regions"""
    # Vanuatu: west of Fiji
    if lon360 >= 165.0 and lon360 <= 172.0 and lat >= -20.0 and lat <= -13.0:
        return 'Vanuatu'
    # Fiji Platform: around center (175-185°E in 360 system)
    elif lon360 >= 175.0 and lon360 <= 185.0 and lat >= -20.0 and lat <= -15.0:
        return 'Fiji'
    # Tonga: east of Fiji (182-187°E in 360 system)
    elif lon360 >= 182.0 and lon360 <= 187.0 and lat >= -23.0 and lat <= -15.0:
        return 'Tonga'
    else:
        return 'Other'

if len(gps_region) > 0:
    gps_region['region'] = gps_region.apply(
        lambda row: classify_region(row.lon360, row.lat), axis=1
    )
    
    print("\n" + "="*70)
    print("REGIONAL STATISTICS - VELOCITIES RELATIVE TO FIJI (0,0)")
    print("="*70)
    for region_name in ['Vanuatu', 'Fiji', 'Tonga']:
        region_data = gps_region[gps_region.region == region_name]
        if len(region_data) > 0:
            mean_vt = region_data.Vt.mean()
            mean_vr = region_data.Vr.mean()
            mean_ve = region_data.Ve.mean()
            mean_vn = region_data.Vn.mean()
            mean_speed = region_data.speed.mean()
            mean_r = region_data.r_km.mean()
            
            print(f"\n{region_name} ({len(region_data)} stations):")
            print(f"  Distance from Fiji: {mean_r:.0f} km")
            print(f"  \n  ABSOLUTE VELOCITIES (GSRM frame):")
            print(f"    Ve (East):  {mean_ve:7.2f} mm/yr")
            print(f"    Vn (North): {mean_vn:7.2f} mm/yr")
            print(f"    Speed:      {mean_speed:7.2f} mm/yr")
            print(f"  \n  RELATIVE TO FIJI (0,0):")
            print(f"    Vr (Radial):      {mean_vr:7.2f} mm/yr", end="")
            if mean_vr > 0:
                print(f" → DIVERGING from Fiji")
            else:
                print(f" → CONVERGING toward Fiji")
            print(f"    Vt (Tangential):  {mean_vt:7.2f} mm/yr", end="")
            if mean_vt > 0:
                print(f" → COUNTERCLOCKWISE")
            else:
                print(f" → CLOCKWISE")
            print(f"  \n  INTERPRETATION:")
            print(f"    Total relative speed: {np.sqrt(mean_vt**2 + mean_vr**2):.2f} mm/yr")

##############################
# 6. VISUALIZATION (FIJI AT ORIGIN)
##############################
if len(gps_region) > 0:
    fig = plt.figure(figsize=(20, 7))
    
    # Panel 1: Geographic view
    ax1 = plt.subplot(131)
    
    for region_name in ['Vanuatu', 'Fiji', 'Tonga']:
        region_data = gps_region[gps_region.region == region_name]
        if len(region_data) > 0:
            color = {'Vanuatu': 'blue', 'Fiji': 'green', 'Tonga': 'red'}[region_name]
            ax1.quiver(
                region_data.lon360, region_data.lat,
                region_data.Ve, region_data.Vn,
                angles='xy', scale_units='xy', scale=1.5,
                color=color, alpha=0.7,
                label=f"{region_name} ({len(region_data)})"
            )
    
    ax1.plot(lon0_360, lat0, 'k*', markersize=25, label='Fiji Center', zorder=10)
    ax1.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latitude (°S)', fontsize=12, fontweight='bold')
    ax1.set_title('GPS Velocities (GSRM Frame)\nGeographic View', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Centered on Fiji with absolute velocities
    ax2 = plt.subplot(132)
    
    for region_name in ['Vanuatu', 'Fiji', 'Tonga']:
        region_data = gps_region[gps_region.region == region_name]
        if len(region_data) > 0:
            color = {'Vanuatu': 'blue', 'Fiji': 'green', 'Tonga': 'red'}[region_name]
            ax2.quiver(
                region_data.x/1000, region_data.y/1000,
                region_data.Ve, region_data.Vn,
                angles='xy', scale_units='xy', scale=1.5,
                color=color, alpha=0.7,
                label=region_name
            )
    
    ax2.plot(0, 0, 'k*', markersize=25, label='Fiji (0,0)', zorder=10)
    ax2.set_xlabel('East-West (km)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('North-South (km)', fontsize=12, fontweight='bold')
    ax2.set_title('Absolute Velocities\n(Fiji at Origin)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Panel 3: Radial vs Tangential (RELATIVE velocities)
    ax3 = plt.subplot(133)
    
    for region_name in ['Vanuatu', 'Fiji', 'Tonga']:
        region_data = gps_region[gps_region.region == region_name]
        if len(region_data) > 0:
            color = {'Vanuatu': 'blue', 'Fiji': 'green', 'Tonga': 'red'}[region_name]
            ax3.scatter(
                region_data.Vr, region_data.Vt,
                c=color, s=150, alpha=0.7,
                label=f"{region_name} (n={len(region_data)})",
                edgecolors='black', linewidth=1.5
            )
    
    ax3.axhline(0, color='k', linestyle='-', alpha=0.5, linewidth=2)
    ax3.axvline(0, color='k', linestyle='-', alpha=0.5, linewidth=2)
    ax3.set_xlabel('Vr (Radial) [mm/yr]\n← Convergent | Divergent →', 
                   fontsize=12, fontweight='bold')
    ax3.set_ylabel('Vt (Tangential) [mm/yr]\n← Clockwise | Counterclockwise →', 
                   fontsize=12, fontweight='bold')
    ax3.set_title('Velocities RELATIVE TO FIJI\n(Fiji = 0,0 with V=0)', 
                  fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Add quadrant labels
    xlim = ax3.get_xlim()
    ylim = ax3.get_ylim()
    ax3.text(xlim[1]*0.7, ylim[1]*0.85, 'CCW\nDivergent', 
             ha='center', fontsize=10, alpha=0.6, fontweight='bold')
    ax3.text(xlim[0]*0.7, ylim[1]*0.85, 'CCW\nConvergent', 
             ha='center', fontsize=10, alpha=0.6, fontweight='bold')
    ax3.text(xlim[0]*0.7, ylim[0]*0.85, 'CW\nConvergent', 
             ha='center', fontsize=10, alpha=0.6, fontweight='bold')
    ax3.text(xlim[1]*0.7, ylim[0]*0.85, 'CW\nDivergent', 
             ha='center', fontsize=10, alpha=0.6, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fiji_normalized_to_origin.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nKey Points:")
    print("  • Fiji is at (0,0) in the Cartesian frame")
    print("  • All velocities are shown relative to Fiji")
    print("  • Fiji itself has zero velocity in this reference frame")
    print("  • Vr/Vt plot shows motion relative to Fiji center")
    print("  • Negative Vr = convergence toward Fiji")
    print("  • Positive Vt = counterclockwise rotation")
else:
    print("\nNo data available for analysis.")