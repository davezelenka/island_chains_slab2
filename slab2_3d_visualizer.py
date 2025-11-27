#!/usr/bin/env python3
"""
Slab 2.0 3D Visualizer

Creates a 3D visualization of up to two subduction slabs with transparent boundaries.
Supports earthquake data filtering (to be implemented for plotting).

Usage:
python slab2_3d_visualizer.py --input_dep1 ker/ker_slab2_dep_02.24.18.grd \
                               --input_thk1 ker/ker_slab2_thk_02.24.18.grd \
                               --input_dep2 van/van_slab2_dep_02.23.18.grd \
                               --input_thk2 van/van_slab2_thk_02.23.18.grd \
                               --e_lon 188.938919 \
                               --w_lon 163.658907 \
                               --n_lat -9.873789 \
                               --s_lat -34.722721 \
                               --earthquakes query_cleaned.csv \
                               --output_plot slab_3d.png \
                               --dpi 300
"""

import argparse
import numpy as np
import pandas as pd
import sys

try:
    import xarray as xr
except ImportError:
    print("Error: xarray is required. Install with: pip install xarray")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available. Install with: pip install plotly")
    print("Falling back to static matplotlib plots")
    PLOTLY_AVAILABLE = False


def load_grd_file(filename):
    """Load a GMT grid file using xarray"""
    try:
        ds = xr.open_dataset(filename)
        return ds
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def get_coordinate_names(dataset):
    """Determine the coordinate names in the dataset"""
    coords = list(dataset.coords.keys())
    dims = list(dataset.dims.keys())
    
    # Common coordinate name variations
    lat_names = ['lat', 'latitude', 'y', 'Y']
    lon_names = ['lon', 'longitude', 'x', 'X']
    
    lat_coord = None
    lon_coord = None
    
    # Find latitude coordinate
    for name in lat_names:
        if name in coords or name in dims:
            lat_coord = name
            break
    
    # Find longitude coordinate
    for name in lon_names:
        if name in coords or name in dims:
            lon_coord = name
            break
    
    # If not found, use the first two coordinates/dimensions
    if lat_coord is None or lon_coord is None:
        available = coords if coords else dims
        if len(available) >= 2:
            lon_coord = available[0]
            lat_coord = available[1]
    
    return lat_coord, lon_coord


def extract_slab_surface(dep_data, thk_data, lon_min, lon_max, lat_min, lat_max):
    """
    Extract slab top and bottom surfaces within the specified bounds
    
    Returns:
    - lon_grid, lat_grid, top_depth_grid, bottom_depth_grid
    """
    # Get coordinate names
    lat_coord, lon_coord = get_coordinate_names(dep_data)
    
    # Get variable names
    dep_var = list(dep_data.data_vars)[0]
    thk_var = list(thk_data.data_vars)[0]
    
    # Get coordinate arrays
    lons = dep_data[lon_coord].values
    lats = dep_data[lat_coord].values
    
    # Handle longitude wrapping (convert to 0-360 if needed)
    if lon_min > 180 or lon_max > 180:
        lons = np.where(lons < 0, lons + 360, lons)
    
    # Filter to region of interest
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    
    # Extract data
    dep_subset = dep_data[dep_var].sel({
        lon_coord: lons[lon_mask],
        lat_coord: lats[lat_mask]
    })
    
    thk_subset = thk_data[thk_var].sel({
        lon_coord: lons[lon_mask],
        lat_coord: lats[lat_mask]
    })
    
    # Create meshgrids
    lon_filtered = lons[lon_mask]
    lat_filtered = lats[lat_mask]
    lon_grid, lat_grid = np.meshgrid(lon_filtered, lat_filtered)
    
    # Get depth values (negative values = depth below surface)
    # Keep the actual negative values from the data
    top_depth = dep_subset.values
    thickness = thk_subset.values
    bottom_depth = top_depth - np.abs(thickness)  # Bottom is deeper (more negative)
    
    print(f"  Data shape: {top_depth.shape}")
    print(f"  Lon range: {lon_filtered.min():.2f} to {lon_filtered.max():.2f}")
    print(f"  Lat range: {lat_filtered.min():.2f} to {lat_filtered.max():.2f}")
    print(f"  Top depth range: {np.nanmin(top_depth):.1f} to {np.nanmax(top_depth):.1f} km")
    print(f"  Bottom depth range: {np.nanmin(bottom_depth):.1f} to {np.nanmax(bottom_depth):.1f} km")
    
    return lon_grid, lat_grid, top_depth, bottom_depth


def load_earthquakes(filename, lon_min, lon_max, lat_min, lat_max):
    """Load and filter earthquake data"""
    try:
        df = pd.read_csv(filename)
        
        # Filter to region
        mask = ((df['lon360'] >= lon_min) & (df['lon360'] <= lon_max) &
                (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max))
        
        filtered_df = df[mask].copy()
        
        print(f"\nEarthquake data loaded:")
        print(f"  Total earthquakes: {len(df)}")
        print(f"  Earthquakes in region: {len(filtered_df)}")
        
        if len(filtered_df) > 0:
            print(f"  Depth range: {filtered_df['depth_km'].min():.1f} to {filtered_df['depth_km'].max():.1f} km")
            print(f"  Magnitude range: {filtered_df['mag'].min():.1f} to {filtered_df['mag'].max():.1f}")
        
        return filtered_df
        
    except Exception as e:
        print(f"Warning: Could not load earthquakes from {filename}: {e}")
        return None


def create_3d_plot(slab1_data, slab2_data, earthquakes_df, 
                   output_file='slab_3d.png', dpi=300, title=None, interactive=True,
                   center_lon=None, center_lat=None):
    """
    Create 3D visualization of slabs
    
    Parameters:
    - slab1_data: tuple of (lon_grid, lat_grid, top_depth, bottom_depth, name) or None
    - slab2_data: tuple of (lon_grid, lat_grid, top_depth, bottom_depth, name) or None
    - earthquakes_df: DataFrame with earthquake data (optional)
    - output_file: Output filename
    - dpi: Resolution
    - title: Plot title
    - interactive: If True and plotly available, create interactive plot
    - center_lon: Longitude of center point to mark
    - center_lat: Latitude of center point to mark
    """
    
    if interactive and PLOTLY_AVAILABLE:
        create_interactive_plot(slab1_data, slab2_data, earthquakes_df, output_file, title,
                               center_lon, center_lat)
    else:
        create_static_plot(slab1_data, slab2_data, earthquakes_df, output_file, dpi, title,
                          center_lon, center_lat)


def create_interactive_plot(slab1_data, slab2_data, earthquakes_df, 
                           output_file='slab_3d.html', title=None,
                           center_lon=None, center_lat=None):
    """Create interactive 3D plot using plotly"""
    
    fig = go.Figure()
    
    # Color scheme for slabs
    colors = ['rgb(255, 107, 107)', 'rgb(78, 205, 196)']  # Red-ish and teal
    alpha = 0.3
    
    slabs = [slab1_data, slab2_data]
    
    # Track depth range for center point
    all_depths = []
    
    for idx, slab_data in enumerate(slabs):
        if slab_data is None:
            continue
        
        lon_grid, lat_grid, top_depth, bottom_depth, name = slab_data
        color = colors[idx]
        
        # Collect depths for center point positioning
        all_depths.extend([np.nanmin(top_depth), np.nanmax(top_depth)])
        
        # Plot top surface
        fig.add_trace(go.Surface(
            x=lon_grid,
            y=lat_grid,
            z=top_depth,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=alpha,
            name=f'{name} - Top',
            hovertemplate='Lon: %{x:.2f}°<br>Lat: %{y:.2f}°<br>Depth: %{z:.1f} km<extra></extra>'
        ))
        
        # Plot bottom surface
        fig.add_trace(go.Surface(
            x=lon_grid,
            y=lat_grid,
            z=bottom_depth,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=alpha * 0.7,
            name=f'{name} - Bottom',
            hovertemplate='Lon: %{x:.2f}°<br>Lat: %{y:.2f}°<br>Depth: %{z:.1f} km<extra></extra>'
        ))
    
    # Add center point marker if specified
    if center_lon is not None and center_lat is not None and len(all_depths) > 0:
        # Place marker at surface (depth = 0)
        center_depth = 0
        
        fig.add_trace(go.Scatter3d(
            x=[center_lon],
            y=[center_lat],
            z=[center_depth],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='diamond'),
            text=['Fiji'],
            textposition='top center',
            name='Fiji',
            hovertemplate=f'Center<br>Lon: {center_lon:.3f}°<br>Lat: {center_lat:.3f}°<extra></extra>'
        ))
        
        # Add vertical line from surface to deepest point
        deepest = min(all_depths)
        fig.add_trace(go.Scatter3d(
            x=[center_lon, center_lon],
            y=[center_lat, center_lat],
            z=[center_depth, deepest],
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        print(f"Center point marked at: ({center_lon:.3f}°, {center_lat:.3f}°)")
    
    # TODO: Plot earthquakes (to be implemented)
    if earthquakes_df is not None and len(earthquakes_df) > 0:
        #print("\nNote: Earthquake plotting not yet implemented")
        # Future implementation:
        fig.add_trace(go.Scatter3d(
             x=earthquakes_df['lon360'],
             y=earthquakes_df['latitude'],
             z=earthquakes_df['depth_km'],
             mode='markers',
             marker=dict(size=earthquakes_df['mag'], color='black', opacity=0.5),
             name='Earthquakes'
        ))
    
    # Set layout
    plot_title = title if title else '3D Slab Visualization (Interactive)'
    
    fig.update_layout(
        title=dict(text=plot_title, x=0.5, xanchor='center', font=dict(size=16)),
        scene=dict(
            xaxis_title='Longitude (°)',
            yaxis_title='Latitude (°)',
            zaxis_title='Depth (km)',
            zaxis=dict(autorange=True),  # Negative values go down
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)  # Initial viewing angle
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1/2)
        ),
        width=1200,
        height=800,
        showlegend=True
    )
    
    # Determine output format based on filename
    if output_file.endswith('.html'):
        fig.write_html(output_file)
        print(f"\nInteractive 3D plot saved to: {output_file}")
        print("Open this file in a web browser to interact with the visualization")
    else:
        # Save as static image (requires kaleido)
        try:
            fig.write_image(output_file)
            print(f"\nStatic image saved to: {output_file}")
        except Exception as e:
            print(f"\nWarning: Could not save as image: {e}")
            html_file = output_file.rsplit('.', 1)[0] + '.html'
            fig.write_html(html_file)
            print(f"Saved as interactive HTML instead: {html_file}")


def create_static_plot(slab1_data, slab2_data, earthquakes_df=None,
                      output_file='slab_3d.png', dpi=300, title=None,
                      center_lon=None, center_lat=None):
    """Create static 3D plot using matplotlib"""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color scheme for slabs
    colors = ['#FF6B6B', '#4ECDC4']  # Red-ish and teal
    alpha = 0.3  # Transparency
    
    slabs = [slab1_data, slab2_data]
    
    # Track depth range for center point
    all_depths = []
    
    for idx, slab_data in enumerate(slabs):
        if slab_data is None:
            continue
        
        lon_grid, lat_grid, top_depth, bottom_depth, name = slab_data
        color = colors[idx]
        
        # Collect depths for center point positioning
        all_depths.extend([np.nanmin(top_depth), np.nanmax(top_depth)])
        
        # Plot top surface
        ax.plot_surface(lon_grid, lat_grid, top_depth,
                       color=color, alpha=alpha, 
                       linewidth=0, antialiased=True,
                       label=f'{name} - Top')
        
        # Plot bottom surface
        ax.plot_surface(lon_grid, lat_grid, bottom_depth,
                       color=color, alpha=alpha * 0.7,
                       linewidth=0, antialiased=True,
                       label=f'{name} - Bottom')
        
        # Add contour lines on top surface for better visualization
        ax.contour(lon_grid, lat_grid, top_depth,
                  levels=10, colors=color, linewidths=0.5, alpha=0.6)
    
    # Add center point marker if specified
    if center_lon is not None and center_lat is not None and len(all_depths) > 0:
        # Place marker at surface (depth = 0)
        center_depth = 0
        
        ax.scatter([center_lon], [center_lat], [center_depth],
                  c='red', s=200, marker='D', edgecolors='darkred',
                  linewidths=2, label='Center Point', zorder=100)
        
        # Add vertical line from surface to deepest point
        deepest = min(all_depths)
        ax.plot([center_lon, center_lon], 
               [center_lat, center_lat],
               [center_depth, deepest],
               'r--', linewidth=2, alpha=0.7)
        
        print(f"Center point marked at: ({center_lon:.3f}°, {center_lat:.3f}°)")
    
    # TODO: Plot earthquakes (to be implemented)
    if earthquakes_df is not None and len(earthquakes_df) > 0:
        print("\nNote: Earthquake plotting not yet implemented")
        # Future implementation:
        # ax.scatter(earthquakes_df['lon360'], 
        #           earthquakes_df['latitude'],
        #           earthquakes_df['depth_km'],
        #           c='black', s=earthquakes_df['mag']*10, alpha=0.5)
    
    # Set labels
    ax.set_xlabel('Longitude (°)', fontsize=11, labelpad=10)
    ax.set_ylabel('Latitude (°)', fontsize=11, labelpad=10)
    ax.set_zlabel('Depth (km)', fontsize=11, labelpad=10)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title('3D Slab Visualization (Static)', fontsize=14, fontweight='bold', pad=20)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Invert z-axis so negative depth values go down
    ax.invert_zaxis()
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"\nStatic 3D plot saved to: {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create 3D visualization of Slab 2.0 data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Slab 1 inputs
    parser.add_argument('--input_dep1', 
                       help='Input depth grid file for slab 1')
    parser.add_argument('--input_thk1',
                       help='Input thickness grid file for slab 1')
    parser.add_argument('--name1', default='Slab 1',
                       help='Name for slab 1 (default: Slab 1)')
    
    # Slab 2 inputs
    parser.add_argument('--input_dep2',
                       help='Input depth grid file for slab 2')
    parser.add_argument('--input_thk2',
                       help='Input thickness grid file for slab 2')
    parser.add_argument('--name2', default='Slab 2',
                       help='Name for slab 2 (default: Slab 2)')
    
    # Geographic bounds
    parser.add_argument('--e_lon', type=float, required=True,
                       help='Eastern longitude bound')
    parser.add_argument('--w_lon', type=float, required=True,
                       help='Western longitude bound')
    parser.add_argument('--n_lat', type=float, required=True,
                       help='Northern latitude bound')
    parser.add_argument('--s_lat', type=float, required=True,
                       help='Southern latitude bound')
    
    # Center point marker
    parser.add_argument('--center_lon', type=float,
                       help='Longitude of center point to mark')
    parser.add_argument('--center_lat', type=float,
                       help='Latitude of center point to mark')
    
    # Earthquake data
    parser.add_argument('--earthquakes',
                       help='CSV file with earthquake data')
    
    # Output options
    parser.add_argument('--output_plot', default='slab_3d.html',
                       help='Output file (default: slab_3d.html for interactive, use .png for static)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for static output plot (default: 300)')
    parser.add_argument('--title',
                       help='Custom title for the plot')
    parser.add_argument('--static', action='store_true',
                       help='Force static matplotlib plot instead of interactive plotly')
    
    args = parser.parse_args()
    
    # Validate that at least one slab is provided
    if not (args.input_dep1 and args.input_thk1) and not (args.input_dep2 and args.input_thk2):
        print("Error: At least one slab (dep and thk files) must be provided")
        sys.exit(1)
    
    # Set up geographic bounds
    lon_min = min(args.w_lon, args.e_lon)
    lon_max = max(args.w_lon, args.e_lon)
    lat_min = min(args.s_lat, args.n_lat)
    lat_max = max(args.s_lat, args.n_lat)
    
    print(f"Geographic bounds:")
    print(f"  Longitude: {lon_min:.3f} to {lon_max:.3f}")
    print(f"  Latitude: {lat_min:.3f} to {lat_max:.3f}")
    
    # Load slab 1
    slab1_data = None
    if args.input_dep1 and args.input_thk1:
        print(f"\nLoading Slab 1 ({args.name1})...")
        dep1 = load_grd_file(args.input_dep1)
        thk1 = load_grd_file(args.input_thk1)
        
        if dep1 is not None and thk1 is not None:
            lon1, lat1, top1, bot1 = extract_slab_surface(
                dep1, thk1, lon_min, lon_max, lat_min, lat_max
            )
            slab1_data = (lon1, lat1, top1, bot1, args.name1)
        else:
            print(f"Warning: Could not load Slab 1 data")
    
    # Load slab 2
    slab2_data = None
    if args.input_dep2 and args.input_thk2:
        print(f"\nLoading Slab 2 ({args.name2})...")
        dep2 = load_grd_file(args.input_dep2)
        thk2 = load_grd_file(args.input_thk2)
        
        if dep2 is not None and thk2 is not None:
            lon2, lat2, top2, bot2 = extract_slab_surface(
                dep2, thk2, lon_min, lon_max, lat_min, lat_max
            )
            slab2_data = (lon2, lat2, top2, bot2, args.name2)
        else:
            print(f"Warning: Could not load Slab 2 data")
    
    # Check if we have at least one slab
    if slab1_data is None and slab2_data is None:
        print("\nError: No valid slab data could be loaded")
        sys.exit(1)
    
    # Load earthquakes if provided
    earthquakes_df = None
    if args.earthquakes:
        earthquakes_df = load_earthquakes(
            args.earthquakes, lon_min, lon_max, lat_min, lat_max
        )
    
    # Create 3D plot
    print("\nCreating 3D visualization...")
    interactive = not args.static and PLOTLY_AVAILABLE
    create_3d_plot(slab1_data, slab2_data, earthquakes_df,
                   args.output_plot, args.dpi, args.title, interactive,
                   args.center_lon, args.center_lat)
    
    print("\n3D visualization completed successfully!")


if __name__ == '__main__':
    main()