"""
High-Precision Python Script for Water Quality and Legal Compliance Analysis in the Jaguaribe River Basin (João Pessoa-PB)

This script uses the Google Earth Engine API and Sentinel-2 data to monitor water quality indices.
"""
import os
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ee
import logging


def main(): 
    """
    Main function to orchestrate the entire workflow.
    """
    # Adicione o ID do seu projeto GEE aqui
    GCP_PROJECT = 'earthengine-project-thesis'

    # Configure o logging básico
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Chame a função de autenticação com o ID do projeto
        authenticate_ee(gcp_project=GCP_PROJECT)
        
        # O restante do seu fluxo de trabalho continua aqui...
        # Ex:
        # roi = load_roi_from_shapefile(SHAPEFILE_PATH)
        # ...
        
    except Exception as e:
        logging.error(f"An error occurred in the main workflow: {e}")

def authenticate_ee(gcp_project: str) -> None:
    """
    Handles GEE authentication and initialization with a specific GCP project.

    Args:
        gcp_project: The Google Cloud Project ID to use for GEE.
    
    Raises:
        Exception: If the initialization fails.
    """
    try:
        # A autenticação (login) é separada da inicialização.
        # ee.Authenticate() abre a janela do navegador e só precisa ser rodada uma vez.
        # ee.Initialize() conecta ao projeto e é necessária em cada execução.
        ee.Initialize(project=gcp_project)
        logging.info(f"Google Earth Engine successfully initialized with project: {gcp_project}")
    except Exception as e:
        logging.error(f"Failed to initialize Earth Engine. Ensure the project ID '{gcp_project}' is correct and that the Earth Engine API is enabled in your Google Cloud Console: https://console.cloud.google.com/apis/library/earthengine.googleapis.com")
        raise e
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- AUTHENTICATION -------------------
def authenticate_ee():
    """Authenticate with Google Earth Engine."""
    logging.info('Authenticating with Earth Engine...')
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

# ------------------- STUDY AREA -------------------
def get_study_area(center_coords=None, buffer_km=5, geojson_path=None):
    """
    Define study area geometry.
    - If geojson_path is provided, import geometry from file.
    - Otherwise, use center_coords and buffer.
    """
    if geojson_path:
        logging.info(f'Importing study area from {geojson_path}...')
        with open(geojson_path) as f:
            geojson = json.load(f)
        return ee.Geometry.Polygon(geojson['features'][0]['geometry']['coordinates'])
    else:
        logging.info(f'Using central point {center_coords} with {buffer_km} km buffer...')
        point = ee.Geometry.Point(center_coords)
        return point.buffer(buffer_km * 1000)

# ------------------- CLOUD MASK -------------------
def mask_clouds(image):
    """
    Apply cloud mask using QA60 band (Sentinel-2).
    """
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask)

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
    # Generates and saves a time series plot and an interactive map of the results.
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
    authenticate_ee()
    # Define study area (default: central point with buffer)
    study_area = get_study_area(center_coords=[-36.133, -7.483], buffer_km=5)
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

# Este bloco agora apenas CHAMA a função que já foi definida acima
if __name__ == '__main__':
    main() # <--- CORRETO: esta linha executa todo o código da função main
