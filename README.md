# Automated Water Quality and Environmental Risk Assessment using Remote Sensing and Google Earth Engine

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## 1. Introduction

The degradation of urban water bodies is a critical environmental issue in Brazil, exacerbated by challenges in continuous, large-scale water quality monitoring. This project introduces a scalable, automated methodology to assess water quality and environmental risk using publicly available satellite data.

By integrating **Google Earth Engine (GEE)** with **Sentinel-2 multispectral imagery**, this research develops a reproducible workflow for calculating key water quality parameters. The goal is to create a tool that can provide systematic, data-driven insights for environmental management, policy enforcement, and academic research.

As an initial application and proof-of-concept, this study focuses on the **Jaguaribe River Basin** in João Pessoa, Paraíba, a water body facing significant environmental pressure from urbanization.

## 2. Project Objectives

* **Develop an Automated Workflow:** Create a fully automated Python script to acquire, process, and analyze satellite imagery for water quality assessment.
* **Estimate Key Water Quality Parameters:** Implement scientifically validated algorithms to derive concentrations of Chlorophyll-a, Total Suspended Solids (TSS), and Turbidity.
* **Assess Legal Compliance:** Programmatically compare estimated parameters against the legal thresholds established by Brazilian environmental law (**CONAMA Resolution 357/2005**).
* **Generate Actionable Outputs:** Produce time-series data, visualizations, and interactive maps to support decision-making and scientific reporting.
* **Establish a Replicable Methodology:** Build a flexible framework that can be adapted to other river basins in the future.

## 3. Case Study: Jaguaribe River Basin, João Pessoa (PB)

The Jaguaribe River Basin was selected as the initial study area due to its ecological significance and the intense environmental impacts it suffers. [cite_start]The basin is entirely located within the urban perimeter of João Pessoa, making it highly vulnerable to anthropogenic pressures[cite: 36].

Key documented issues include:
* [cite_start]Disorderly urban occupation, leading to the removal of riparian vegetation and landfilling of banks[cite: 12, 113].
* [cite_start]Pollution from domestic sewage and solid waste disposal, contributing to eutrophication and degradation of water quality[cite: 112, 114, 125, 127].
* [cite_start]Increased soil impermeabilization, which intensifies surface runoff, erosion, and the silting of the river channel[cite: 71, 74, 77].

This context makes the Jaguaribe River an ideal candidate for testing and validating the automated monitoring methodology developed in this project.

## 4. Methodology

The workflow is executed entirely within a Python environment, leveraging the Google Earth Engine API for cloud-based geospatial processing.

#### 4.1. Data Acquisition and Preprocessing
* **Satellite Platform:** Sentinel-2 MSI (Level-2A Surface Reflectance).
* **Temporal Range:** 2018 to present.
* **Cloud Masking:** Images are filtered using the Sentinel-2 Scene Classification Layer (SCL) to remove pixels corresponding to clouds, cloud shadows, and other artifacts.
* **Water Body Masking:** A dynamic water mask is generated using the Normalized Difference Water Index (NDWI) to ensure that analysis is restricted to water surfaces.
    * $NDWI = \frac{(Green - NIR)}{(Green + NIR)}$

#### 4.2. Water Quality Parameter Estimation
The following indices are calculated for each valid satellite image:
* [cite_start]**Chlorophyll-a ($Chl_a$):** An indicator of phytoplankton biomass and eutrophication[cite: 458]. Estimated using an adaptation of the OC2 algorithm for Sentinel-2 bands.
* **Total Suspended Solids (TSS):** Measures the total amount of solid material suspended in the water, affecting clarity. Calculated using the semi-empirical model proposed by Nechad et al. (2010).
* **Turbidity:** A measure of water cloudiness or haziness. Estimated using an empirical model based on the reflectance of the Red band (B4).

#### 4.3. Legal and Environmental Framework (Enquadramento)
The results are evaluated against the standards set by **CONAMA Resolution 357/2005** for **Class 2 freshwater bodies**, which is the classification applicable to the Jaguaribe River. The key thresholds used for this analysis are:
* [cite_start]**Chlorophyll-a:** $\le 30 \ \mu g/L$ [cite: 557]
* **Turbidity:** $\le 100 \ NTU$ (Nephelometric Turbidity Units)

#### 4.4. Temporal Analysis and Aggregation
The script aggregates the pixel values within the delineated watershed for each date and calculates the monthly mean for each water quality parameter. A compliance flag is generated for each month, indicating whether the parameters are within the legal limits.

## 5. Technologies & Tools
* **Programming Language:** Python 3.10
* **Geospatial Processing:** Google Earth Engine (GEE), Geopandas
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Folium
* **Environment Management:** Conda

## 6. Workflow and Outputs

The execution of the main script (`src/main.py`) triggers the complete workflow:
1.  Authentication with GEE.
2.  Delineation of the Jaguaribe River watershed.
3.  Fetching and preprocessing of the Sentinel-2 image collection.
4.  Calculation of water quality indices for each image.
5.  Monthly aggregation and compliance analysis.
6.  Generation of the final outputs.

The script produces the following files in the `outputs/` directory:
* `report_jaguaribe.csv`: A CSV file containing monthly time-series data for each index, including compliance status.
* `timeseries_jaguaribe.png`: A plot showing the temporal variation of water quality indices with CONAMA 357/2005 legal limits marked for reference.
* `map_jaguaribe.html`: An interactive Folium map displaying the delineated watershed.

## 7. Installation and Usage

#### 7.1. Prerequisites
* A Google Earth Engine account.
* Google Cloud SDK initialized and authenticated.
* Conda/Miniconda installed.

#### 7.2. Setup
1.  **Clone the repository:**
    ```bash
    git clone [URL-DO-SEU-REPOSITORIO]
    cd [NOME-DO-SEU-REPOSITORIO]
    ```
2.  **Create and activate the Conda environment:**
    ```bash
    conda create -n jaguaribe-env python=3.10
    conda activate jaguaribe-env
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

#### 7.3. Running the Analysis
1.  Open the `src/main.py` script and set your Google Cloud Project ID in the `GCP_PROJECT` variable.
2.  Execute the script from the root directory of the project:
    ```bash
    python src/main.py
    ```

## 8. Future Work
This project serves as a foundation for a broader research initiative. Future development will focus on:
* **Scientific Validation:** Correlating satellite-derived estimates with *in-situ* water quality measurements.
* **Advanced Risk Modeling:** Developing a multi-criteria environmental risk index that incorporates additional variables (e.g., land use, population density, pollution sources).
* **Expansion to Other Basins:** Applying and adapting the methodology to monitor other critical watersheds in Brazil.
* **Web Application:** Creating an interactive web-based dashboard for visualizing results and allowing non-technical users to explore the data.

## 9. References
* CONAMA Resolution 357/2005. *Dispõe sobre a classificação dos corpos de água e diretrizes ambientais para o seu enquadramento*.
* McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing, 17*(7), 1425-1432.
* Nechad, B., Ruddick, K., & Neukermans, G. (2010). Calibration and validation of a generic multisensor algorithm for mapping of turbidity in coastal waters. *Remote Sensing of Environment, 114*(4), 854-866.
* Santos, C. L. et al. (2016). Impactos da Urbanização em Bacias Hidrográficas: O Caso da Bacia do Rio Jaguaribe, Cidade de João Pessoa/PB. *REGNE, 2*.
* Sentinel-2 User Guide. European Space Agency (ESA).

---
*For questions, contributions, or to cite this work, please contact Jaidna Dantas de Almeida.*
