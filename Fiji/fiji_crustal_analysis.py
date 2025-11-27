"""
Fiji-centered GNSS velocity check (rotated GSRM psvelo or CSV)
Save as: fiji_gsrm_check.py

Requires: numpy, pandas, matplotlib, scipy, requests (requests only for remote download)
Optional: cartopy for map plotting.

Usage examples:
python fiji_gsrm_check.py --url "https://gsrm.unavco.org/..." --center_lat -18.1248 --center_lon 178.4501
or
python fiji_gsrm_check.py --input velocities.psvelo --center_lat -18.1248 --center_lon 178.4501
"""

import argparse
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sqrt
from scipy.interpolate import griddata

# Try import requests (for downloading); if not present, instruct user to download manually
try:
    import requests
except Exception:
    requests = None

def parse_psvelo(text):
    """
    Parse a GMT psvelo-like file into a DataFrame with columns:
    lon, lat, vx(mm/yr east), vy(mm/yr north), [optional others]
    This parser handles typical layout: lon lat ve vn se sn corr (and comment lines starting with # or >)
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('>') or line.startswith('!'):
            continue
        parts = line.split()
        # Need at least 4 numeric columns: lon lat ve vn
        numeric = []
        for p in parts:
            try:
                numeric.append(float(p))
            except:
                numeric.append(np.nan)
        if len(numeric) >= 4:
            lon, lat, ve, vn = numeric[0], numeric[1], numeric[2], numeric[3]
            rows.append((lon, lat, ve, vn))
    df = pd.DataFrame(rows, columns=['lon','lat','vx_mmyr','vy_mmyr'])
    # convert longitudes to -180..180
    df['lon'] = ((df['lon'] + 180) % 360) - 180
    return df

def parse_csv(text):
    # Try read_csv from text buffer (pandas will infer)
    df = pd.read_csv(io.StringIO(text))
    # Expect columns named: lon, lat, vx_mmyr, vy_mmyr or similar
    # try to normalize common names
    cols = {c.lower():c for c in df.columns}
    mapping = {}
    if 'lon' in cols:
        mapping[cols['lon']] = 'lon'
    elif 'longitude' in cols:
        mapping[cols['longitude']] = 'lon'
    if 'lat' in cols:
        mapping[cols['lat']] = 'lat'
    elif 'latitude' in cols:
        mapping[cols['latitude']] = 'lat'
    if 'vx' in cols:
        mapping[cols['vx']] = 'vx_mmyr'
    if 'vy' in cols:
        mapping[cols['vy']] = 'vy_mmyr'
    if 've' in cols:
        mapping[cols['ve']] = 'vx_mmyr'
    if 'vn' in cols:
        mapping[cols['vn']] = 'vy_mmyr'
    if 'east' in cols:
        mapping[cols['east']] = 'vx_mmyr'
    if 'north' in cols:
        mapping[cols['north']] = 'vy_mmyr'
    df = df.rename(columns=mapping)
    if not {'lon','lat','vx_mmyr','vy_mmyr'}.issubset(df.columns):
        raise ValueError("CSV did not contain required columns after mapping. Columns found: " + ",".join(df.columns))
    df['lon'] = ((df['lon'] + 180) % 360) - 180
    return df[['lon','lat','vx_mmyr','vy_mmyr']]

def load_remote_or_local(path_or_url):
    if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
        if requests is None:
            raise RuntimeError("requests library not available. Download the file manually and provide --input local_path.")
        r = requests.get(path_or_url)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} for {path_or_url}")
        text = r.text
    else:
        with open(path_or_url,'r', encoding='utf-8') as f:
            text = f.read()
    # Heuristic: if file contains many lines with at least 4 numbers and no comma => psvelo; else CSV
    sample = "\n".join(text.splitlines()[:40])
    if ',' in sample:
        df = parse_csv(text)
    else:
        # try psvelo parse
        df = parse_psvelo(text)
        # fall back to CSV if not many rows
        if len(df) < 5:
            try:
                df = parse_csv(text)
            except Exception as e:
                pass
    return df

def compute_radial(df, center_lat, center_lon):
    # approximate local ENU using degree->km conversion
    deg_to_km = 111.32
    dy_km = (df['lat'].values - center_lat) * deg_to_km
    dx_km = (df['lon'].values - center_lon) * deg_to_km * np.cos(np.radians(0.5*(df['lat'].values + center_lat)))
    r = np.sqrt(dx_km**2 + dy_km**2)
    r[r==0] = np.nan
    ux = dx_km / r
    uy = dy_km / r
    # velocities mm/yr to km/yr
    vx_kmyr = df['vx_mmyr'].values * 1e-6
    vy_kmyr = df['vy_mmyr'].values * 1e-6
    v_rad_kmyr = vx_kmyr * ux + vy_kmyr * uy
    v_tan_kmyr = -vx_kmyr * uy + vy_kmyr * ux
    df['v_rad_mmyr'] = v_rad_kmyr * 1e6
    df['v_tan_mmyr'] = v_tan_kmyr * 1e6
    df['speed_mmyr'] = np.sqrt(df['vx_mmyr']**2 + df['vy_mmyr']**2)
    return df

def plot_vectors(df, center_lat, center_lon, title="Fiji-centered velocities"):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
    except Exception:
        use_cartopy = False

    lons = df['lon'].values
    lats = df['lat'].values
    vx = df['vx_mmyr'].values
    vy = df['vy_mmyr'].values
    vr = df['v_rad_mmyr'].values

    fig = plt.figure(figsize=(10,8))
    if use_cartopy:
        ax = plt.axes(projection=ccrs.PlateCarree())
        extent = [center_lon-20, center_lon+20, center_lat-15, center_lat+15]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND)
        ax.gridlines(draw_labels=True)
        ax.scatter(lons, lats, s=8, transform=ccrs.PlateCarree(), zorder=5)
        # convert mm/yr to degrees for plotting only
        mm_to_deg = 1.0 / 111320000.0
        u = vx * mm_to_deg * 200.0
        v = vy * mm_to_deg * 200.0
        for xi, yi, ui, vi, vrv in zip(lons, lats, u, v, vr):
            col = 'red' if vrv > 0 else 'blue'
            ax.arrow(xi, yi, ui, vi, transform=ccrs.PlateCarree(),
                     head_width=0.2, head_length=0.25, fc=col, ec=col, alpha=0.7)
        ax.plot(center_lon, center_lat, 'k*', markersize=12, transform=ccrs.PlateCarree())
    else:
        ax = plt.subplot(111)
        ax.scatter(lons, lats, s=8)
        mm_to_deg = 1.0 / 111320000.0
        u = vx * mm_to_deg * 200.0
        v = vy * mm_to_deg * 200.0
        for xi, yi, ui, vi, vrv in zip(lons, lats, u, v, vr):
            col = 'red' if vrv > 0 else 'blue'
            ax.arrow(xi, yi, ui, vi, head_width=0.2, head_length=0.25, fc=col, ec=col, alpha=0.7)
        ax.plot(center_lon, center_lat, 'k*', markersize=12)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    plt.title(title + "\n(red outward radial; blue inward)")
    plt.show()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', help='URL to psvelo or CSV file', default=None)
    p.add_argument('--input', help='Local input file (psvelo or CSV)', default=None)
    p.add_argument('--center_lat', type=float, default=-18.1248, help='Fiji center latitude')
    p.add_argument('--center_lon', type=float, default=178.4501, help='Fiji center longitude')
    args = p.parse_args()

    if not args.url and not args.input:
        print("Provide --url or --input (file). See header comments.")
        return

    path = args.url if args.url else args.input
    df = load_remote_or_local(path)
    if df is None or len(df) == 0:
        print("No stations parsed from file.")
        return

    df = compute_radial(df, args.center_lat, args.center_lon)

    print("Stations:", len(df))
    frac_out = (df['v_rad_mmyr'] > 0).sum() / len(df)
    print(f"Fraction outward from center: {frac_out:.2%}")
    print("Mean radial (mm/yr):", df['v_rad_mmyr'].mean())
    print("Top outward stations:")
    print(df.sort_values('v_rad_mmyr', ascending=False).head(10)[['lon','lat','v_rad_mmyr','speed_mmyr']])
    print("Top inward stations:")
    print(df.sort_values('v_rad_mmyr').head(10)[['lon','lat','v_rad_mmyr','speed_mmyr']])

    plot_vectors(df, args.center_lat, args.center_lon)

if __name__ == '__main__':
    main()
