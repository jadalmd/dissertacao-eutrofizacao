"""
Fully Automated Eutrophication Risk Assessment for Epitácio Pessoa Reservoir (Boqueirão-PB)
Master's Thesis Project

This script uses the Google Earth Engine API and Sentinel-2 data to monitor water quality indices (NDCI, TSS, Zsd).
"""

import ee
import os
import sys
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# ------------------- CONFIGURATION -------------------
logging.basicConfig(level=logging.INFO, format='%(message)s')

OUTPUT_DIR = 'outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ------------------- AUTHENTICATION -------------------
def authenticate_ee(gcp_project):
    """
    Handles GEE authentication and initialization.
    """
    logging.info('Authenticating with Google Earth Engine...')
    try:
        ee.Authenticate()
        ee.Initialize(project=gcp_project)
        logging.info('Earth Engine successfully initialized.')
    except Exception as e:
        logging.error(f'Error during Earth Engine initialization: {e}')
        sys.exit(1)

# ------------------- WATERSHED DELINEATION -------------------
def delineate_watershed(poi):
    """
    Automatically delineates the watershed using SRTM DEM and ee.Terrain.watershed().
    """
    logging.info('Delineating watershed for the study area...')
    dem = ee.Image('USGS/SRTMGL1_003')
    watershed = ee.Terrain.watershed(dem, poi)
    return watershed

# ------------------- WATER & CLOUD MASKING -------------------
def mask_sentinel2_water(image):
    """
    Robust water and cloud/shadow mask using SCL and NDWI.
    """
    scl = image.select('SCL')
    # SCL mask: water (6), exclude clouds (8,9), cloud shadow (3), snow/ice (11)
    water_mask = scl.eq(6)
    cloud_mask = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(3)).Or(scl.eq(11))
    # NDWI mask
    ndwi = image.normalizedDifference(['B3', 'B8'])
    ndwi_mask = ndwi.gt(0.1)
    final_mask = water_mask.And(ndwi_mask).And(cloud_mask.Not())
    return image.updateMask(final_mask)

# ------------------- INDEX CALCULATIONS -------------------
def calculate_indices(image):
    """
    Calculates NDCI, TSS, Zsd indices.
    """
    # NDCI
    ndci = image.normalizedDifference(['B5', 'B4']).rename('ndci')
    # TSS
    tss = image.expression(
        '(496.09 * B4) / (1 - (B4 / 0.22))', {
            'B4': image.select('B4')
        }).rename('tss')
    # Zsd
    zsd = image.expression(
        '1.48 * (B2 / B3) + 0.12', {
            'B2': image.select('B2'),
            'B3': image.select('B3')
        }).rename('zsd')
    return image.addBands([ndci, tss, zsd])

# ------------------- IMAGE COLLECTION PROCESSING -------------------
def process_image_collection(start_date, end_date, roi):
    """
    Filters Sentinel-2, applies mask, calculates indices, aggregates monthly.
    """
    logging.info('Processing image collection...')
    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate(start_date, end_date)
                  .filterBounds(roi)
                  .map(mask_sentinel2_water)
                  .map(calculate_indices))
    # Monthly aggregation
    def monthly_mean(date):
        monthly = collection.filterDate(date, date.advance(1, 'month')).mean()
        return monthly.set({'date': date.format('YYYY-MM')})
    # List of months
    months = pd.date_range(start=start_date, end=end_date, freq='MS')
    monthly_images = [monthly_mean(ee.Date(str(m))) for m in months]
    return ee.ImageCollection(ee.List(monthly_images))

# ------------------- EXPORT RESULTS -------------------
def export_results(monthly_collection, roi, start_date, end_date):
    """
    Exports monthly mean indices to CSV and GeoJSON.
    """
    logging.info('Exporting data to CSV and GeoJSON...')
    # Sample points at centroid
    centroid = roi.centroid()
    features = []
    for img in tqdm(monthly_collection.toList(monthly_collection.size()).getInfo()):
        image = ee.Image(img['id']) if 'id' in img else None
        if image:
            date = image.get('date').getInfo() if image.get('date') else None
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=10,
                maxPixels=1e9
            ).getInfo()
            stats['date'] = date
            features.append(stats)
    df = pd.DataFrame(features)
    df = df.dropna(subset=['ndci', 'tss', 'zsd'])
    csv_path = os.path.join(OUTPUT_DIR, f'indices_boqueirao_{start_date}_{end_date}.csv')
    geojson_path = os.path.join(OUTPUT_DIR, f'indices_boqueirao_{start_date}_{end_date}.geojson')
    df.to_csv(csv_path, index=False)
    df.to_json(geojson_path, orient='records', date_format='iso')
    return df

# ------------------- VISUALIZATION -------------------
def create_visualizations(df, roi_geojson_path):
    """
    Generates time series plot and interactive Folium map.
    """
    logging.info('Generating visualizations...')
    # Time series plot
    plt.figure(figsize=(12, 8))
    for col in ['ndci', 'tss', 'zsd']:
        plt.plot(df['date'], df[col], label=col.upper())
    plt.title('Water Quality Indices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'timeseries_plot.png')
    plt.savefig(plot_path)
    plt.close()
    # Folium map
    gdf = gpd.read_file(roi_geojson_path)
    centroid = gdf.geometry.centroid.iloc[0]
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12)
    folium.GeoJson(gdf, name='Reservoir', style_function=lambda x: {'color': 'blue', 'weight': 2}).add_to(m)
    # Add NDCI markers
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=6,
            popup=f"Date: {row['date']}<br>NDCI: {row['ndci']:.2f}",
            color='green', fill=True, fill_opacity=0.7
        ).add_to(m)
    map_path = os.path.join(OUTPUT_DIR, 'map_boqueirao.html')
    m.save(map_path)

# ------------------- MAIN EXECUTION -------------------
def main():
    try:
        # User must set their GCP project ID here
        GCP_PROJECT = 'earthengine-project-thesis'
        authenticate_ee(GCP_PROJECT)
        # Define POI (centroid of reservoir)
        poi = ee.Geometry.Point([-36.13, -7.48])
        watershed = delineate_watershed(poi)
        START_DATE = '2023-01-01'
        END_DATE = '2024-12-31'
        monthly_collection = process_image_collection(START_DATE, END_DATE, watershed)
        df = export_results(monthly_collection, watershed, START_DATE, END_DATE)
        # Save watershed as GeoJSON for visualization
        geojson_path = os.path.join(OUTPUT_DIR, 'boqueirao_waterbody.geojson')
        # Export watershed geometry to GeoJSON
        coords = watershed.coordinates().getInfo()
        gdf = gpd.GeoDataFrame({'geometry': [gpd.GeoSeries.from_wkt(str(ee.Geometry.Polygon(coords).toWkt()))]})
        gdf.to_file(geojson_path, driver='GeoJSON')
        create_visualizations(df, geojson_path)
        logging.info('Script completed successfully.')
    except Exception as e:
        logging.error(f'Error in main workflow: {e}')

if __name__ == '__main__':
    main()

# ------------------- MASKING FUNCTION -------------------
def mask_sentinel2_water(image):
    # Use Scene Classification Layer (SCL) to mask water pixels (value 6)
    scl = image.select('SCL')
    water_mask = scl.eq(6)
    return image.updateMask(water_mask)

# ------------------- INDEX CALCULATIONS -------------------
def calculate_ndci(image):
    # NDCI = (B5 - B4) / (B5 + B4)
    # Reference: Mishra & Mishra (2012)
    ndci = image.normalizedDifference(['B5', 'B4']).rename('NDCI')
    return image.addBands(ndci)

def calculate_tss(image):
    # TSS = (A_rho * B4) / (1 - (B4 / C_rho)), A_rho=496.09, C_rho=0.22
    # Reference: Nechad et al. (2010)
    tss = image.expression(
        '(A * rho) / (1 - (rho / C))', {
            'rho': image.select('B4'),
            'A': 496.09,
            'C': 0.22
        }).rename('TSS')
    return image.addBands(tss)

def calculate_zsd(image):
    # Zsd = 1.48 * (B2 / B3) + 0.12
    # Reference: Page et al. (2019)
    zsd = image.expression(
        '1.48 * (BLUE / GREEN) + 0.12', {
            'BLUE': image.select('B2'),
            'GREEN': image.select('B3')
        }).rename('Zsd')
    return image.addBands(zsd)

# ------------------- IMAGE COLLECTION PROCESSING -------------------
def process_collection(roi):
    logging.info('Loading and processing Sentinel-2 image collection...')
    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                 .filterDate(START_DATE, END_DATE)
                 .filterBounds(roi)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
                 .map(mask_sentinel2_water)
                 .map(calculate_ndci)
                 .map(calculate_tss)
                 .map(calculate_zsd))
    return collection

# ------------------- MONTHLY MEANS -------------------
def calculate_monthly_mean(collection, roi):
    logging.info('Calculating monthly means...')
    def monthly_stats(img):
        date = ee.Date(img.get('system:time_start')).format('YYYY-MM').getInfo()
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=10,
            maxPixels=1e9
        )
        stats = stats.getInfo()
        stats['date'] = date
        return stats
    image_list = collection.toList(collection.size())
    results = []
    for i in range(image_list.size().getInfo()):
        img = ee.Image(image_list.get(i))
        try:
            stats = monthly_stats(img)
            results.append(stats)
        except Exception as e:
            logging.warning(f'Error processing image {i}: {e}')
    df = pd.DataFrame(results)
    df = df.dropna(subset=['NDCI', 'TSS', 'Zsd'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby(df['date'].dt.to_period('M')).mean().reset_index()
    df['date'] = df['date'].dt.to_timestamp()
    return df

# ------------------- EXPORT TO CSV & GEOJSON -------------------
def export_results(df):
    csv_path = os.path.join(OUTPUT_DIR, f'indices_boqueirao_{START_DATE}_{END_DATE}.csv')
    geojson_path = os.path.join(OUTPUT_DIR, f'indices_boqueirao_{START_DATE}_{END_DATE}.geojson')
    logging.info(f'Exporting CSV to {csv_path}...')
    df.to_csv(csv_path, index=False)
    logging.info(f'Exporting GeoJSON to {geojson_path}...')
    df.to_json(geojson_path, orient='records', date_format='iso')

# ------------------- TIME SERIES PLOTS -------------------
def plot_time_series(df):
    logging.info('Generating time series plots...')
    plt.figure(figsize=(12, 8))
    for col in ['NDCI', 'TSS', 'Zsd']:
        sns.lineplot(x='date', y=col, data=df, label=col)
    plt.title('Water Quality Indices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'timeseries_plot.png')
    plt.savefig(plot_path)
    logging.info(f'Time series plot saved to {plot_path}')
    plt.close()

# ------------------- INTERACTIVE MAP -------------------
def create_folium_map(df, roi, collection):
    logging.info('Creating interactive Folium map...')
    # Select most recent, least cloudy image
    image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
    center = roi.centroid().coordinates().getInfo()
    m = folium.Map(location=center, zoom_start=12)
    # Add NDCI, TSS, Zsd as overlays (dummy markers for demonstration)
    if not df.empty:
        row = df.iloc[-1]
        popup_text = (f"NDCI: {row['NDCI']:.2f}<br>"
                      f"TSS: {row['TSS']:.2f}<br>"
                      f"Zsd: {row['Zsd']:.2f}")
        folium.Marker(location=center, popup=popup_text).add_to(m)
    map_path = os.path.join(OUTPUT_DIR, 'map_boqueirao.html')
    m.save(map_path)
    logging.info(f'Folium map saved to {map_path}')

# ------------------- MAIN WORKFLOW -------------------
def main():
    gcp_project = 'earthengine-project-thesis'
    authenticate_ee(gcp_project)
    roi = get_roi()
    collection = process_collection(roi)
    df = calculate_monthly_mean(collection, roi)
    export_results(df)
    plot_time_series(df)
    create_folium_map(df, roi, collection)
    logging.info('Script completed.')

if __name__ == '__main__':
    main()

"""
Water Quality Monitoring Script for Epitácio Pessoa Reservoir (Boqueirão, Paraíba, Brazil)
Master's Thesis Project

This script uses the Google Earth Engine API and Sentinel-2 data to monitor water quality indices.
"""

import ee
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
from datetime import datetime

# ------------------- CONFIGURATION -------------------
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Define global variables for date range (10-year period)
START_DATE = '2015-01-01'  # Change as needed
END_DATE = '2024-12-31'    # Change as needed

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- AUTHENTICATION -------------------
def authenticate_ee():
    """Authenticate with Google Earth Engine."""
    logging.info('Authenticating with Earth Engine...')
    try:
        ee.Initialize(project='earthengine-project-thesis')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='earthengine-project-thesis')

# ------------------- CLOUD MASK -------------------
def mask_clouds(image):
    """
    Apply cloud mask using QA60 band (Sentinel-2).
    """
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

# ------------------- MNDWI & VECTORIZE -------------------
def calc_mndwi_and_vectorize():
    """
    Calculate MNDWI and vectorize water body for study area.
    MNDWI = (Green - SWIR) / (Green + SWIR)
    Uses max water extent over 10 years.
    """
    logging.info('Calculating MNDWI and vectorizing water body...')
    # 1. Define a preliminary bounding box around the reservoir to limit the processing area.
    # The coordinates are for the Boqueirão reservoir region.
    preliminary_geometry = ee.Geometry.Polygon([
        [-36.21, -7.53],
        [-36.21, -7.42],
        [-36.05, -7.42],
        [-36.05, -7.53],
        [-36.21, -7.53]
    ])

    # Load Sentinel-2 SR collection for 10 years
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR')
          .filterDate(START_DATE, END_DATE)
          # 2. Filter the collection by the preliminary bounding box.
          .filterBounds(preliminary_geometry)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))

    # Select bands
    def add_mndwi(img):
        green = img.select('B3')
        swir = img.select('B11')
        mndwi = green.subtract(swir).divide(green.add(swir)).rename('MNDWI')
        return img.addBands(mndwi)

    s2 = s2.map(add_mndwi)

    # Max water extent (max MNDWI composite)
    mndwi_max = s2.select('MNDWI').max()

    # Threshold to get water mask
    water_mask = mndwi_max.gt(0.3)

    # Vectorize water body
    vectors = water_mask.selfMask().reduceToVectors(
        # 3. Pass the preliminary geometry to the function.
        geometry=preliminary_geometry,
        scale=10,
        geometryType='polygon',
        eightConnected=True,
        maxPixels=1e9
    )

    # The rest of the code remains the same.
    # Get largest polygon (reservoir)
    def get_largest(feats):
        largest = None
        max_area = 0
        for f in feats:
            area = ee.Feature(f).geometry().area(maxError=10).getInfo()
            if area > max_area:
                largest = f
                max_area = area
        return largest

    # Download features to client
    features = vectors.getInfo()['features']
    largest = get_largest(features)
    study_geom = ee.Geometry.Polygon(largest['geometry']['coordinates'])
    return study_geom

# ------------------- WATER QUALITY INDICES -------------------
def calc_chla(image):
    """
    Calculate Chlorophyll-a (Chl-a) using formula:
    Chl-a = 21 * (B8 / B4) - 5
    Reference: prompt.txt
    """
    b8 = image.select('B8')
    b4 = image.select('B4')
    chla = b8.divide(b4).multiply(21).subtract(5).rename('Chl_a')
    return image.addBands(chla)

def calc_tss(image):
    """
    Calculate Turbidity (TSS) using Nechad 2010:
    TSS = 228.1 * (B4 / (1 - (B4 / 0.1686))) + 1.18
    Reference: prompt.txt
    """
    b4 = image.select('B4')
    tss = b4.divide(ee.Image(1).subtract(b4.divide(0.1686))).multiply(228.1).add(1.18).rename('TSS')
    return image.addBands(tss)

def calc_secchi(image):
    """
    Calculate Secchi Disk depth (Zsd):
    Zsd = 8.24 * log(B3 / B2) + 10.6
    Reference: prompt.txt
    """
    b3 = image.select('B3')
    b2 = image.select('B2')
    zsd = b3.divide(b2).log().multiply(8.24).add(10.6).rename('Secchi')
    return image.addBands(zsd)

# ------------------- DATA PROCESSING -------------------
def process_images(study_area):
    """
    Load Sentinel-2 SR images, apply cloud mask, clip, and calculate indices.
    """
    logging.info('Loading Sentinel-2 SR collection...')
    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate(START_DATE, END_DATE)
                  .filterBounds(study_area)
                  .map(mask_clouds)
                  .map(lambda img: img.clip(study_area))
                  .map(calc_chla)
                  .map(calc_tss)
                  .map(calc_secchi))
    return collection

# ------------------- EXPORT RESULTS -------------------
def export_results(df, start_date, end_date):
    """
    Export monthly mean results as CSV and JSON to /outputs/.
    """
    filename_base = f'indices_boqueirao_{start_date}_{end_date}'
    csv_path = os.path.join(OUTPUT_DIR, filename_base + '.csv')
    json_path = os.path.join(OUTPUT_DIR, filename_base + '.json')
    logging.info(f'Exporting CSV to {csv_path}...')
    df.to_csv(csv_path, index=False)
    logging.info(f'Exporting JSON to {json_path}...')
    df.to_json(json_path, orient='records', date_format='iso')

# ------------------- PLOT TIME SERIES -------------------
def plot_timeseries(df):
    """
    Generate time series plots for each index.
    """
    plt.figure(figsize=(12, 8))
    for col in ['Chl_a', 'TSS', 'Secchi']:
        sns.lineplot(x='date', y=col, data=df, label=col)
    plt.title('Water Quality Indices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'timeseries_plot.png')
    plt.savefig(plot_path)
    logging.info(f'Time series plot saved to {plot_path}')
    plt.close()

# ------------------- FOLIUM MAP -------------------
def create_folium_map(df, study_area, selected_date):
    """
    Create interactive Folium map showing mean values for selected date.
    """
    logging.info(f'Creating Folium map for {selected_date}...')
    # Get mean values for selected date
    row = df[df['date'] == selected_date].iloc[0]
    center = study_area.centroid().coordinates().getInfo()
    m = folium.Map(location=center, zoom_start=12)
    popup_text = (f"Chl-a: {row['Chl_a']:.2f}<br>"
                  f"TSS: {row['TSS']:.2f}<br>"
                  f"Secchi: {row['Secchi']:.2f}")
    folium.Marker(location=center, popup=popup_text).add_to(m)
    map_path = os.path.join(OUTPUT_DIR, f'folium_map_{selected_date}.html')
    m.save(map_path)
    logging.info(f'Folium map saved to {map_path}')

# ------------------- MAIN WORKFLOW -------------------
def main():
    gcp_project = 'earthengine-project-thesis'
    authenticate_ee(gcp_project)
    # Calculate study area using MNDWI and vectorization
    study_area = calc_mndwi_and_vectorize()
    collection = process_images(study_area)

    # Reduce images to monthly means
    def monthly_means(img):
        date = ee.Date(img.get('system:time_start')).format('YYYY-MM').getInfo()
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=study_area,
            scale=10,
            maxPixels=1e9
        )
        stats = stats.getInfo()
        stats['date'] = date
        return stats

    logging.info('Calculating monthly means...')
    image_list = collection.toList(collection.size())
    results = []
    for i in range(image_list.size().getInfo()):
        img = ee.Image(image_list.get(i))
        try:
            stats = monthly_means(img)
            results.append(stats)
        except Exception as e:
            logging.warning(f'Error processing image {i}: {e}')

    df = pd.DataFrame(results)
    df = df.dropna(subset=['Chl_a', 'TSS', 'Secchi'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby(df['date'].dt.to_period('M')).mean().reset_index()
    df['date'] = df['date'].dt.to_timestamp()

    export_results(df, START_DATE, END_DATE)
    plot_timeseries(df)
    # Create Folium map for the first available date
    if not df.empty:
        create_folium_map(df, study_area, df['date'].iloc[0].strftime('%Y-%m-%d'))
    logging.info('Script completed.')

if __name__ == '__main__':
    main()
