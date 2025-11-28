# Fiji-Tonga-Kermadec Subduction Geometry Analysis

**Arc Geometry as a Diagnostic of Subduction Mechanism**

This project analyzes the geometric constraints on subduction mechanisms in the Fiji-Tonga-Kermadec system, testing whether arc curvature provides diagnostic evidence for compression-driven override versus slab-pull driven subduction.

## Overview

Standard subduction models predict convex or straight arc geometry, but observations show systematically concave arcs. This project:

1. Tests whether observed arcs follow logarithmic spiral geometry
2. Analyzes GPS velocity data to quantify convergence vs rotation
3. Examines the relationship between arc curvature and convergence rate
4. Visualizes 3D slab geometry and seismicity patterns

**Key Hypothesis:** Arc curvature κ should correlate with convergence rate v under compression-driven override:
```
κ ∝ v_convergence
```

For Fiji-Tonga-Kermadec: κ_Tonga / κ_Vanuatu ≈ 3.3 (matching velocity ratio 147/45 mm/yr)

## Project Structure

```
fiji-subduction-analysis/
├── README.md
├── log_spiral_regression.py      # Fit KML arc data to logarithmic spirals
├── analyze_fiji_vortex.py         # GPS velocity analysis and decomposition
├── fiji_crustal_analysis.py       # Strain rate and vorticity analysis
├── fiji_gsrm_check.py             # GSRM data validation
├── slab2_3d_visualizer.py         # 3D visualization of slab geometry
├── data/
│   ├── GPS_PA.gmt                 # Pacific plate GPS velocities
│   ├── GPS_TO.gmt                 # Tonga microplate GPS velocities
│   ├── GPS_vectors_after_rotation_PA.dat
│   ├── velocity_PA.dat            # GSRM velocity model
│   ├── vel_0.1deg_AU.gmt          # Australian plate velocity grid
│   ├── vel_0.1deg_PA.gmt          # Pacific plate velocity grid
│   ├── fiji_vortex_analysis.csv   # Computed vortex metrics
│   ├── ker_spiral.kml             # Kermadec arc spiral coordinates
│   └── ton_spiral.kml             # Tonga arc spiral coordinates
└── docs/
    ├── arc_geometry_paper.tex     # LaTeX manuscript
    └── references.bib             # Bibliography
```

## Installation

### Requirements

```bash
pip install numpy pandas scipy matplotlib xarray plotly
```

### Optional (for 3D visualization)
```bash
pip install kaleido  # For static image export from plotly
```

## Data Acquisition

Download required GSRM data files to the `data/` directory:

```bash
mkdir -p data
cd data

# GPS station velocities
wget https://geodesy.unr.edu/GSRM/GPS_PA.gmt
wget https://geodesy.unr.edu/GSRM/GPS_TO.gmt

# GSRM velocity models
wget https://gsrm.unavco.org/model/files/1.2/velocity_PA.dat
wget https://gsrm.unavco.org/model/files/1.2/GPS_vectors_after_rotation_PA.dat

# Velocity grids
wget https://geodesy.unr.edu/GSRM/vel_0.1deg_AU.gmt
wget https://geodesy.unr.edu/GSRM/vel_0.1deg_PA.gmt
```

### KML Arc Coordinates

Extract arc coordinates from Google Earth:
- **Project KML**: https://earth.google.com/earth/d/1DdjB9jU_wtnr_xpSC2MIt9p0bmUqC2nw?usp=sharing
- Export `ker_spiral.kml` (Kermadec arc)
- Export `ton_spiral.kml` (Tonga arc)
- Place in `data/` directory

## Usage

### 1. Logarithmic Spiral Regression

Test whether arc geometry follows logarithmic spiral form: r(θ) = r₀ exp(k·θ)

```bash
python log_spiral_regression.py
```

**Output:**
- R² goodness of fit
- Spiral tightness parameter `k`
- Visualization: `log_spiral_fit.png`

**Expected Results:**
- Vanuatu: k ≈ 0.067 (loose spiral)
- Tonga: k ≈ 0.165 (tighter spiral)
- Ratio: k_Tonga / k_Vanuatu ≈ 2.5

### 2. GPS Velocity Analysis

Decompose GPS velocities into radial (convergence) and tangential (rotation) components:

```bash
python analyze_fiji_vortex.py
```

**Output:**
- Velocity decomposition table
- Convergence rates (V_r)
- Tangential velocities (V_t)
- Spiral tightness ratios

**Expected Results:**
- Tonga: V_r = -145 mm/yr, V_t = +24 mm/yr
- Vanuatu: V_r = -33 mm/yr, V_t = -2 mm/yr
- System is 85-93% convergent

### 3. Strain Rate and Vorticity Analysis

Compute spatial gradients of velocity field:

```bash
python fiji_crustal_analysis.py
```

**Output:**
- Strain rate tensor fields
- Vorticity distribution
- Maximum compression orientation

### 4. 3D Slab Visualization

Visualize slab geometry with earthquake distributions:

```bash
python slab2_3d_visualizer.py \
    --input_dep1 ker/ker_slab2_dep_02.24.18.grd \
    --input_thk1 ker/ker_slab2_thk_02.24.18.grd \
    --input_dep2 van/van_slab2_dep_02.23.18.grd \
    --input_thk2 van/van_slab2_thk_02.23.18.grd \
    --e_lon 188.94 --w_lon 163.66 \
    --n_lat -9.87 --s_lat -34.72 \
    --center_lat -17.75 --center_lon 178.0 \
    --earthquakes query_cleaned.csv
```

**Output:**
- Interactive 3D HTML visualization
- Slab surfaces (top and bottom)
- Earthquake distributions
- Center point marker

**Note:** Slab 2.0 grid files must be obtained separately from USGS.

## Key Results

### Geometric Predictions

| Arc | Convergence Rate | Predicted k | Observed k | R² |
|-----|------------------|-------------|------------|-----|
| Tonga | 147 mm/yr | 0.165 | [TBD] | [TBD] |
| Vanuatu | 45 mm/yr | 0.067 | [TBD] | [TBD] |
| Ratio | 3.3× | 2.5× | [TBD] | — |

### GPS Velocity Decomposition

| Region | V_r (mm/yr) | V_t (mm/yr) | \|V_t\|/\|V_r\| | Interpretation |
|--------|-------------|-------------|-----------------|----------------|
| Vanuatu | -33.37 | -2.23 | 0.067 | Convergent + weak CW |
| Fiji | 6.82 | -0.61 | 0.089 | Squeezed |
| Tonga | -145.02 | +23.85 | 0.165 | Convergent + CCW |

**System is 85-93% convergent (compression-dominated), only 7-15% rotational.**

### Seismicity Patterns

| Depth Range | Interface % | Center % | Interpretation |
|-------------|-------------|----------|----------------|
| Shallow (25-70 km) | 68-82% | 18-32% | Mechanical coupling |
| Intermediate (70-300 km) | Variable | Variable | Transition zone |
| Deep (>300 km) | 1-9% | 91-95% | Internal deformation |

Depth-dependent transition supports compression-induced slab buckling.

## Theoretical Framework

### Curvature Evolution Equation

Arc shape evolution under forcing velocity v_n:

```
∂κ/∂t = -∂²v_n/∂s² - κ² v_n
```

**Slab Pull Prediction:** Center retreats fastest → ∂²v_n/∂s² < 0 → κ < 0 (convex)

**Compression Prediction:** Center advances fastest → ∂²v_n/∂s² > 0 → κ > 0 (concave)

### Phase Diagram

Arc geometry determined by dimensionless parameter:

```
Λ = σ_H / (Δρ g h L_s)
```

- Λ >> 1: Compression dominates → concave arcs
- Λ << 1: Slab pull dominates → convex arcs
- Λ ≈ 1: Transitional geometry

**For Tonga:** Λ ≈ 3-5 → compression-dominated → concave geometry required

## Citations

### Key References

- **Martin (2014)**: Toroidal flows in North Fiji Basin (12-10 Ma to 1.5 Ma)
- **Mahadevan et al. (2010)**: Energy minimization in arc curvature
- **Schellart et al. (2010)**: Toroidal flow and trench rollback
- **Kreemer et al. (2014)**: GSRM velocity model
- **Jarrard (1986)**: Global arc curvature observations

See `docs/references.bib` for complete bibliography.

## Contributing

This is an active research project. Contributions welcome:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Data Sources

- **GSRM**: Global Strain Rate Map (https://gsrm.unavco.org/)
- **Slab 2.0**: USGS subduction zone geometry (https://www.sciencebase.gov/catalog/item/5aa1b00ee4b0b1c392e86467)
- **USGS Earthquake Catalog**: https://earthquake.usgs.gov/earthquakes/search/

## Acknowledgments

- UNAVCO for GSRM data
- USGS for Slab 2.0 models
- Community for feedback and discussion

---

**Status:** Pre-publication research project
