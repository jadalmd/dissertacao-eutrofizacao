# Water Quality and Legal Compliance Analysis in the Jaguaribe River Basin (João Pessoa-PB)

## Overview
This project provides a fully automated, research-grade Python workflow for monitoring water quality in the Jaguaribe River Basin, João Pessoa, PB, Brazil. It uses Google Earth Engine (GEE) and Sentinel-2 satellite data to generate time-series analyses and assess compliance with Brazilian environmental legislation (CONAMA Resolution 357/2005, Class 2 rivers).

## Installation
1. Create a Conda environment:
   ```bash
   conda create -n jaguaribe-env python=3.10
   conda activate jaguaribe-env
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Edit `src/main.py` to set your Google Cloud Project ID (`GCP_PROJECT`).
2. Run the main script:
   ```bash
   python src/main.py
   ```
3. Outputs will be saved in the `outputs/` directory:
   - `report_jaguaribe.csv`: Monthly water quality indices and compliance report
   - `timeseries_jaguaribe.png`: Time series plot of indices and legal limits
   - `map_jaguaribe.html`: Interactive map of the basin

## Ancillary File Generation
To generate the requirements.txt and README.md files, use:
```bash
pip freeze > requirements.txt
# Edit README.md as needed
```


## Methodology


### Study Area Delineation
The Region of Interest (ROI) for this study, corresponding to the Jaguaribe River watershed, is programmatically delineated using hydrological algorithms available within the Google Earth Engine platform. The delineation uses the ee.Algorithms.Hydro.watershed algorithm, based on an outlet point at the river's mouth (Latitude: -7.1436, Longitude: -34.8256), and the Shuttle Radar Topography Mission (SRTM) 30m Digital Elevation Model (DEM) data (USGS/SRTMGL1_003). This method was chosen for its scientific reproducibility and precision. The resulting polygon serves as the precise boundary for all subsequent analysis in this research.

- **Satellite Data:** Uses Sentinel-2 SR images from GEE, applies cloud/water masking (SCL and NDWI).
- **Indices Calculated:**
   - **Chlorophyll-a (Chl_a):** OC2 algorithm (Sentinel-2 adaptation)
   - **Total Suspended Solids (TSS):** Nechad et al. (2010)
   - **Turbidity:** Empirical model using band B4 (Red)
- **Monthly Aggregation:** Computes monthly means for each index within the ROI.
- **Legal Compliance:** Compares results to CONAMA 357/2005 Class 2 limits.
- **Visualization:** Generates time series plots and an interactive folium map.

## Legal Framework
- **Enquadramento:** Brazilian system for classifying water bodies by primary use.
- **Class 2 Rivers (CONAMA 357/2005):**
  - Chlorophyll-a ≤ 30 µg/L
  - Turbidity ≤ 100 NTU
  - (Other parameters can be added as needed)

## Outputs
- `report_jaguaribe.csv`: Monthly indices and compliance flags
- `timeseries_jaguaribe.png`: Time series plot with legal thresholds
- `map_jaguaribe.html`: Interactive map of the basin

## References
- CONAMA Resolution 357/2005: [Link](https://www.mma.gov.br/port/conama/legiabre.cfm?codlegi=459)
- McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. International Journal of Remote Sensing, 17(7), 1425-1432.
- Nechad, B., Ruddick, K., & Neukermans, G. (2010). Calibration and validation of a generic multisensor algorithm for mapping of turbidity in coastal waters. Remote Sensing of Environment, 114(4), 854-866.
- Sentinel-2 User Guide: [Link](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi)

---
For questions or contributions, please contact the project maintainer.
