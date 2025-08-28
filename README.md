# Water Quality Monitoring in Epitácio Pessoa Reservoir (Boqueirão, Paraíba, Brazil)

## Project Description
This project provides a robust Python script for monitoring water quality in the Epitácio Pessoa Reservoir using Google Earth Engine and Sentinel-2 satellite data. It calculates water quality indices (Chlorophyll-a, Turbidity, Secchi Disk depth), generates time series plots, exports results, and creates interactive maps for visualization.

## Installation Instructions

### 1. Python Version
- Requires Python 3.11 or newer.

### 2. Required Libraries
Install the following libraries in your Conda or virtual environment:

```
pip install earthengine-api pandas numpy matplotlib seaborn folium plotly tqdm requests
```

#### List of Required Libraries
- earthengine-api
- pandas
- numpy
- matplotlib
- seaborn
- folium
- plotly
- tqdm
- requests

### 3. Earth Engine Authentication
Run the following command in your terminal to authenticate with Google Earth Engine:

```
python -c "import ee; ee.Authenticate()"
```
Follow the instructions to complete authentication.

## How to Define START_DATE and END_DATE
Edit the following lines in `src/main.py` to set your desired analysis period (default is 10 years):

```python
START_DATE = '2015-01-01'  # Change as needed
END_DATE = '2024-12-31'    # Change as needed
```

## How to Run the Script
1. Ensure all dependencies are installed and Earth Engine is authenticated.
2. Run the script from the project root:

```
python src/main.py
```

## Visualizing Results
- **Time Series Plots:** Saved as `outputs/timeseries_plot.png`.
- **Monthly Mean Results:** Saved as CSV and JSON in the `outputs/` folder, e.g., `indices_boqueirao_2015-01-01_2024-12-31.csv`.
- **Interactive Map:** HTML file for the first available date, e.g., `outputs/folium_map_YYYY-MM-DD.html`.

## Accessing Generated Files
All output files are saved in the `/outputs/` directory. You can open CSV/JSON files in Excel, GIS software, or web mapping platforms.

## Example: Using Raw GitHub URLs for Web Mapping
If your repository is public, you can use the raw file URLs to connect CSV/JSON files to web mapping platforms:

```
https://raw.githubusercontent.com/<username>/<repo>/main/outputs/indices_boqueirao_2015-01-01_2024-12-31.csv
```
Replace `<username>` and `<repo>` with your GitHub details.

---

For questions or troubleshooting, please refer to the script comments and logging messages, or contact the project author.
