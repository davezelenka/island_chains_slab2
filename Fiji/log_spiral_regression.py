#!/usr/bin/env python3
"""
Logarithmic Spiral Regression for Subduction Arc Geometry

Fits KML coordinate data to a logarithmic spiral of the form:
    r(θ) = r₀ * exp(k * θ)

where:
    r = distance from center
    θ = angle from reference direction
    r₀ = initial radius
    k = spiral tightness parameter (related to curvature)

Usage:
    python log_spiral_regression_fixed.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import pearsonr

def parse_kml_coordinates(coord_string):
    """
    Parse KML coordinate string into lon, lat arrays
    
    Format: "lon,lat,alt lon,lat,alt ..."
    """
    coords = coord_string.strip().split()
    lons = []
    lats = []
    
    for coord in coords:
        parts = coord.split(',')
        if len(parts) >= 2:
            lons.append(float(parts[0]))
            lats.append(float(parts[1]))
    
    lons = np.array(lons)
    lats = np.array(lats)
    
    # Convert negative longitudes to 360° format
    lons[lons < 0] += 360
    
    return lons, lats


def cartesian_to_polar(lons, lats, lon_center, lat_center):
    """
    Convert lon/lat to polar coordinates (r, θ) relative to center
    """
    # Convert to Cartesian (simplified, assuming small region)
    x = (lons - lon_center) * np.cos(np.radians(lat_center))
    y = lats - lat_center
    
    # Convert to polar
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Unwrap theta to ensure continuity
    theta = np.unwrap(theta)
    
    return r, theta, x, y


def log_spiral(theta, r0, k, theta0):
    """
    Logarithmic spiral function: r(θ) = r₀ * exp(k * (θ - θ₀))
    
    Parameters:
        theta: angle
        r0: initial radius
        k: spiral tightness (curvature parameter)
        theta0: reference angle offset
    """
    return r0 * np.exp(k * (theta - theta0))


def residuals_log_spiral(params, theta, r_observed):
    """
    Residuals for least squares fitting
    """
    r0, k, theta0 = params
    r_predicted = log_spiral(theta, r0, k, theta0)
    return r_observed - r_predicted


def fit_log_spiral(theta, r):
    """
    Fit logarithmic spiral to polar coordinates using least squares
    
    Returns:
        r0: initial radius
        k: spiral tightness parameter
        theta0: reference angle
        r_squared: coefficient of determination
    """
    # Initial guess
    r0_init = np.mean(r)
    k_init = 0.0  # Start with circular (k=0)
    theta0_init = theta[0]
    
    # Fit using least squares
    result = least_squares(
        residuals_log_spiral,
        [r0_init, k_init, theta0_init],
        args=(theta, r),
        method='lm'
    )
    
    r0_fit, k_fit, theta0_fit = result.x
    
    # Calculate R²
    r_predicted = log_spiral(theta, r0_fit, k_fit, theta0_fit)
    
    # R² = 1 - (SS_res / SS_tot)
    ss_res = np.sum((r - r_predicted)**2)
    ss_tot = np.sum((r - np.mean(r))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Alternative: Pearson correlation coefficient
    pearson_r, _ = pearsonr(r, r_predicted)
    pearson_r_squared = pearson_r**2
    
    return r0_fit, k_fit, theta0_fit, r_squared, pearson_r_squared, r_predicted


def plot_results(lons, lats, lon_center, lat_center, theta, r, r_predicted, 
                r0, k, theta0, r_squared):
    """
    Create visualization of the spiral fit
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Geographic view
    ax1 = axes[0]
    
    # Plot original arc
    ax1.plot(lons, lats, 'bo-', label='Observed arc', markersize=8, linewidth=2)
    
    # Plot fitted spiral
    theta_dense = np.linspace(theta.min(), theta.max(), 200)
    r_dense = log_spiral(theta_dense, r0, k, theta0)
    
    # Convert back to geographic coordinates
    x_dense = r_dense * np.cos(theta_dense)
    y_dense = r_dense * np.sin(theta_dense)
    lon_dense = lon_center + x_dense / np.cos(np.radians(lat_center))
    lat_dense = lat_center + y_dense
    
    ax1.plot(lon_dense, lat_dense, 'r--', label='Fitted log spiral', linewidth=2)
    ax1.plot(lon_center, lat_center, 'g*', markersize=15, label='Center')
    
    ax1.set_xlabel('Longitude (°)', fontsize=12)
    ax1.set_ylabel('Latitude (°)', fontsize=12)
    ax1.set_title(f'Geographic View\nR² = {r_squared:.4f}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Polar view
    ax2 = axes[1]
    
    # Sort by theta for clean plotting
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]
    r_sorted = r[sort_idx]
    r_pred_sorted = r_predicted[sort_idx]
    
    ax2.plot(theta_sorted, r_sorted, 'bo-', label='Observed', markersize=8, linewidth=2)
    ax2.plot(theta_sorted, r_pred_sorted, 'r--', label='Fitted', linewidth=2)
    
    ax2.set_xlabel('Angle θ (radians)', fontsize=12)
    ax2.set_ylabel('Radius r (degrees)', fontsize=12)
    ax2.set_title(f'Polar Coordinates\nk = {k:.4f}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('log_spiral_fit.png', dpi=300, bbox_inches='tight')
    print("Plot saved to: log_spiral_fit.png")
    
    plt.show()


def print_results(lon_center, lat_center, r0, k, theta0, r_squared, pearson_r2):
    """
    Print formatted results
    """
    print("\n" + "="*60)
    print("LOGARITHMIC SPIRAL REGRESSION RESULTS")
    print("="*60)
    
    print(f"\nCenter coordinates (USER SPECIFIED):")
    print(f"  Longitude: {lon_center:.6f}°")
    print(f"  Latitude:  {lat_center:.6f}°")
    
    print(f"\nSpiral parameters:")
    print(f"  r₀ (initial radius):     {r0:.6f}°")
    print(f"  k (spiral tightness):    {k:.6f}")
    print(f"  θ₀ (reference angle):    {theta0:.6f} rad ({np.degrees(theta0):.2f}°)")
    
    print(f"\nGoodness of fit:")
    print(f"  R² (coefficient of determination):  {r_squared:.6f}")
    print(f"  R² (Pearson correlation squared):   {pearson_r2:.6f}")
    
    # Interpret k value
    print(f"\nInterpretation:")
    if abs(k) < 0.01:
        print(f"  k ≈ 0: Nearly circular arc")
    elif k > 0:
        print(f"  k > 0: Outward spiral (radius increases with angle)")
    else:
        print(f"  k < 0: Inward spiral (radius decreases with angle)")
    
    # Spiral type classification
    spiral_tightness = abs(k)
    if spiral_tightness < 0.05:
        print(f"  Classification: Very loose spiral (almost circular)")
    elif spiral_tightness < 0.15:
        print(f"  Classification: Loose spiral")
    elif spiral_tightness < 0.30:
        print(f"  Classification: Moderate spiral")
    else:
        print(f"  Classification: Tight spiral")
    
    # Quality assessment
    print(f"\nQuality assessment:")
    if r_squared > 0.95:
        print(f"  Excellent fit (R² > 0.95)")
    elif r_squared > 0.90:
        print(f"  Very good fit (R² > 0.90)")
    elif r_squared > 0.80:
        print(f"  Good fit (R² > 0.80)")
    elif r_squared > 0.70:
        print(f"  Moderate fit (R² > 0.70)")
    else:
        print(f"  Poor fit (R² < 0.70) - Consider checking center coordinates")
    
    print("\n" + "="*60)


def main():
    # ========== USER-SPECIFIED CENTER ==========
    # Center coordinates (lat, lon format as provided)
    lat_center = -17.699918
    lon_center = 178.053277
    # ===========================================
    
    # Input data (KML coordinates)
    coord_string = """
    178.424777385615,-18.38813843671104,0
    177.7501475355268,-18.45489907919885,0
    177.1078206385478,-18.1972350973003,0
    176.8353904014674,-17.49810616470128,0
    176.917977715173,-16.89738846206183,0
    177.7021187398858,-16.47956866932078,0
    178.5091096875231,-16.08903703422084,0
    179.3870732814122,-15.71009802333254,0
    -179.759197994172,-15.70955363272824,0
    -178.8161874770706,-15.98409910438555,0
    -178.3329974721211,-16.79524104457797,0
    -178.0312077405652,-18.50258908908235,0
    -177.8962370224306,-19.64268344461386,0
    -177.8366205213115,-20.51053787562727,0
    -177.9162851662334,-21.4367725723495,0
    -178.2023197160462,-22.82138614334733,0
    -178.2923441483688,-24.13310515432545,0
    -178.5877975673712,-25.68207773351643,0
    -178.7703122728221,-27.0784360197,0
    """
    
    print("Parsing coordinates...")
    lons, lats = parse_kml_coordinates(coord_string)
    print(f"Parsed {len(lons)} coordinate points")
    
    print(f"\nUsing specified center:")
    print(f"  Latitude:  {lat_center:.6f}°")
    print(f"  Longitude: {lon_center:.6f}°")
    
    # Convert to polar coordinates
    print("\nConverting to polar coordinates...")
    r, theta, x, y = cartesian_to_polar(lons, lats, lon_center, lat_center)
    
    # Fit logarithmic spiral
    print("Fitting logarithmic spiral...")
    r0, k, theta0, r_squared, pearson_r2, r_predicted = fit_log_spiral(theta, r)
    
    # Print results
    print_results(lon_center, lat_center, r0, k, theta0, r_squared, pearson_r2)
    
    # Plot results
    print("\nGenerating visualization...")
    plot_results(lons, lats, lon_center, lat_center, theta, r, r_predicted,
                r0, k, theta0, r_squared)


if __name__ == '__main__':
    main()