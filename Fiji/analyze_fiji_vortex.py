import numpy as np
import pandas as pd
from pathlib import Path

##############################################
# USER SETTINGS
##############################################

DATA_DIR = Path("data/")

# Geographic window you provided
E_LON = 188.938919
W_LON = 163.658907
N_LAT = -9.873789
S_LAT = -34.722721

CENTER_LAT = -17.75
CENTER_LON = 178.0  # vortex center

# EPSG:4326 radius of Earth approximation
R_EARTH = 6371e3


##############################################
# UTILITY FUNCTIONS
##############################################

def read_generic_table(path):
    """
    Attempt to read a whitespace-delimited table.
    Automatically handles GMT-style and basic DAT formats.
    """
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    # Skip header/commented lines starting with #, >
    data = [ln for ln in lines if not ln.lstrip().startswith(("#", ">"))]

    # Split by whitespace
    df = pd.DataFrame([ln.split() for ln in data])

    # Try to convert to numeric when possible
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


def geographic_filter(df, lon_col, lat_col):
    """Select points inside Fiji bounding box."""
    mask = (
        (df[lon_col] >= W_LON) &
        (df[lon_col] <= E_LON) &
        (df[lat_col] >= S_LAT) &
        (df[lat_col] <= N_LAT)
    )
    return df[mask]


def lon_wrap(lon):
    """Wrap longitudes to 0–360 for Pacific-centered analysis."""
    lon = np.asarray(lon)
    lon = np.mod(lon, 360)
    return lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance in meters."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat/2)**2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def local_cartesian(lat, lon, lat0, lon0):
    """
    Convert lat/lon to local tangent plane coordinates (meters)
    centered on (lat0, lon0).
    """
    x = haversine_distance(lat0, lon0, lat0, lon) * np.sign(lon - lon0)
    y = haversine_distance(lat0, lon0, lat, lon0) * np.sign(lat - lat0)
    return x, y


##############################################
# VORTEX DIAGNOSTICS
##############################################

def decompose_velocity(x, y, u, v):
    """
    Decompose velocity vector into:
      - tangential (v_t), orthogonal to radial direction
      - radial (v_r)
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Unit vectors
    e_r_x = np.cos(theta)
    e_r_y = np.sin(theta)
    e_t_x = -np.sin(theta)
    e_t_y = np.cos(theta)

    v_r = u * e_r_x + v * e_r_y
    v_t = u * e_t_x + v * e_t_y
    return r, theta, v_r, v_t


def vortex_score(v_r, v_t):
    """
    Simple vortex indicator: strong tangential, weak radial.
    Returns ratio |v_t| / (|v_r| + epsilon).
    """
    return np.abs(v_t) / (np.abs(v_r) + 1e-6)


##############################################
# MAIN PIPELINE
##############################################

def process_velocity_file(path, lon_col=0, lat_col=1, vx_col=2, vy_col=3):
    """
    Generic reader for velocity files:
    Columns assumed:
      lon  lat  vx(mm/yr)  vy(mm/yr)
    Modify indexes if needed.
    """
    df = read_generic_table(path)

    df[lon_col] = lon_wrap(df[lon_col])
    df = geographic_filter(df, lon_col, lat_col)
    if df.empty:
        return None

    # Convert column names
    df = df.rename(columns={
        lon_col: "lon",
        lat_col: "lat",
        vx_col: "vx",
        vy_col: "vy"
    })

    # Convert mm/yr → m/yr
    df["vx"] = df["vx"].astype(float) / 1000.0
    df["vy"] = df["vy"].astype(float) / 1000.0

    # Local coordinates
    X, Y = local_cartesian(df["lat"], df["lon"], CENTER_LAT, CENTER_LON)
    df["x"] = X
    df["y"] = Y

    # Decompose into radial/tangential
    r, th, v_r, v_t = decompose_velocity(X, Y, df["vx"], df["vy"])
    df["r"] = r
    df["theta"] = th
    df["v_r"] = v_r
    df["v_t"] = v_t
    df["vortex_score"] = vortex_score(v_r, v_t)

    return df


def main():
    print("\n=== Loading & Processing Fiji Region Velocity Data ===\n")

    files = [
        "GPS_PA.gmt",
        "GPS_TO.gmt",
        "GPS_vectors_after_rotation_PA.dat",
        "velocity_PA.dat",
        "vel_0.1deg_AU.gmt",
        "vel_0.1deg_PA.gmt",
    ]

    all_frames = []

    for fname in files:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"[missing] {fname}")
            continue

        print(f"Processing {fname} ...")

        df = process_velocity_file(fpath)
        if df is None or df.empty:
            print(f"  -> No points inside Fiji window.")
            continue

        print(f"  -> {len(df)} points in Fiji region.")
        all_frames.append(df)

    if not all_frames:
        print("\nNo data found in Fiji region for any dataset.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    print("\n=== VORTEX ANALYSIS SUMMARY ===")
    print(f"Total Fiji datapoints: {len(combined)}")

    # Global statistics
    print("\nTangential velocity stats (m/yr):")
    print(combined["v_t"].describe())

    print("\nRadial velocity stats (m/yr):")
    print(combined["v_r"].describe())

    print("\nVortex score statistics:")
    print(combined["vortex_score"].describe())

    # Are most velocities vortex-like?
    mean_vs = combined["vortex_score"].mean()
    pct_strong = np.mean(combined["vortex_score"] > 2.0) * 100

    print(f"\nMean vortex score: {mean_vs:.3f}")
    print(f"Percent strongly tangential (>2): {pct_strong:.2f}%")

    # Save intermediate results
    outpath = DATA_DIR / "fiji_vortex_analysis.csv"
    combined.to_csv(outpath, index=False)
    print(f"\nSaved combined Fiji-region vortex dataset to:\n{outpath}\n")


if __name__ == "__main__":
    main()
