# ==============================================================================
# SCRIPT COMPLETO E CORRIGIDO PARA ANÁLISE DA BACIA DO RIO JAGUARIBE
# ==============================================================================
import ee
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm

# ==============================================================================
# CONSTANTES GLOBAIS E CONFIGURAÇÃO
# ==============================================================================
# ID do Projeto no Google Cloud (substituir se necessário)
GCP_PROJECT = 'earthengine-project-thesis'

# Caminhos (relativos ao diretório /src)
OUTPUT_DIR = '../outputs'

# Período de Análise
START_DATE = '2010-01-01'
END_DATE = '2024-01-01'

# ==============================================================================
# FUNÇÕES DO WORKFLOW
# ==============================================================================

def setup_logging():
    """Configura o sistema de logging para o script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def authenticate_ee(gcp_project: str):
    """Realiza a autenticação e inicialização do Google Earth Engine."""
    try:
        ee.Initialize(project=gcp_project)
        logging.info(f"Google Earth Engine inicializado com sucesso no projeto: {gcp_project}")
    except Exception as e:
        logging.error(f"Falha ao inicializar o Earth Engine. Verifique o ID do projeto e se a API está ativa.")
        raise e

def delineate_watershed_from_outlet(outlet_point: ee.Geometry.Point) -> ee.Geometry:
    """Delineia a bacia hidrográfica a partir de um ponto de exutório usando a metodologia correta da API."""
    logging.info("Iniciando delineamento da bacia hidrográfica...")
    # 1. Carregar o Modelo Digital de Elevação (MDE)
    dem = ee.Image('USGS/SRTMGL1_003')
    # 2. Calcular a direção do fluxo a partir do MDE
    flow_direction = ee.Terrain.flowDirections(dem)
    # 3. Criar uma 'semente' no ponto de saída (exutório)
    from_point = ee.Image(0).byte().paint(outlet_point, 1)
    # 4. Delinear a bacia
    watershed_raster = flow_direction.watershed(from_point)
    # 5. Converter o raster da bacia para um polígono vetorial
    watershed_geom = watershed_raster.toByte().geometry()
    logging.info("Delineamento da bacia concluído.")
    return watershed_geom

def mask_sentinel2_water(image: ee.Image) -> ee.Image:
    """Aplica uma máscara de nuvens e água robusta usando a banda SCL do Sentinel-2."""
    scl = image.select('SCL')
    # Mantém apenas pixels de água (classe 6) e remove nuvens, sombras, etc.
    water_mask = scl.eq(6)
    return image.updateMask(water_mask)

def calculate_indices(image: ee.Image) -> ee.Image:
    """Calcula todos os índices de qualidade da água e os adiciona como bandas."""
    # Clorofila-a (Chl_a)
    chl_a = image.expression(
        '2.1171 + 1.68 * log10(NIR / RED)',
        {'NIR': image.select('B8'), 'RED': image.select('B4')}
    ).rename('Chl_a')

    # Sólidos Suspensos Totais (TSS)
    tss = image.expression(
        '(A * RED) / (1 - (RED / C))',
        {'A': 496.09, 'C': 0.22, 'RED': image.select('B4')}
    ).rename('TSS')

    # Turbidez (Turbidity)
    turbidity = image.expression(
        '186.99 * RED', {'RED': image.select('B4')}
    ).rename('Turbidity')

    # Índice de Estado Trófico (TSI) - CORRIGIDO para usar operações do EE
    tsi = chl_a.log().multiply(-0.6).add(-0.7).divide(ee.Number(2).log()).multiply(-1).add(6).multiply(10).rename('TSI')

    return image.addBands([chl_a, tss, turbidity, tsi])


def generate_monthly_timeseries(roi: ee.Geometry, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Gera uma série temporal mensal de forma eficiente, usando mapeamento no lado do servidor.
    """
    logging.info("Gerando série temporal mensal...")
    
    image_collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                        .filterBounds(roi)
                        .filterDate(start_date, end_date))

    # Gera uma lista de meses no período
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    n_months = end.difference(start, 'month').round()
    months = ee.List.sequence(0, n_months.subtract(1)).map(lambda n: start.advance(n, 'month'))

    # Função para processar cada mês
    def process_month(date):
        date = ee.Date(date)
        # Filtra a coleção para o mês atual
        monthly_coll = image_collection.filterDate(date, date.advance(1, 'month'))
        # Cria uma imagem média para o mês, aplicando todas as máscaras e cálculos
        monthly_image = (monthly_coll
                         .map(mask_sentinel2_water)
                         .map(calculate_indices)
                         .mean()
                         .clip(roi))
        
        # Calcula a média espacial dos índices dentro da ROI
        stats = monthly_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=30,
            maxPixels=1e9
        )
        # Retorna um Feature com as estatísticas e a data
        return ee.Feature(None, stats).set('system:time_start', date.millis())

    # Mapeia a função sobre a lista de meses
    monthly_stats_fc = ee.FeatureCollection(months.map(process_month))
    
    # Extrai as informações para o cliente (seu computador)
    logging.info("Extraindo dados do servidor GEE...")
    features = monthly_stats_fc.getInfo()['features']
    
    # Converte para DataFrame
    results = []
    for f in features:
        props = f['properties']
        # Converte o timestamp para data
        props['date'] = pd.to_datetime(props['system:time_start'], unit='ms')
        results.append(props)
        
    df = pd.DataFrame(results)
    logging.info("Série temporal gerada com sucesso.")
    return df

# =================== FUNÇÕES ERI E CRITÉRIOS ===================
def get_population_density(roi):
    """Calcula a densidade populacional média na ROI usando WorldPop."""
    pop_img = ee.ImageCollection('WORLDPOP/GP/100m/pop').sort('system:time_start', False).first()
    pop_img = pop_img.clip(roi)
    stats = pop_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=100,
        maxPixels=1e9
    )
    try:
        mean_pop = stats.getInfo()['population']
    except Exception:
        mean_pop = None
    return mean_pop

def get_land_use_risk(roi):
    """Calcula o score de risco de uso do solo usando MapBiomas."""
    mb_img = ee.ImageCollection('projects/mapbiomas-workspace/public/collection7/mapbiomas-collection70-brazil').sort('year', False).first()
    mb_img = mb_img.clip(roi)
    # Reclassificação
    high_risk = mb_img.remap([24, 30], [1, 1], 0)
    medium_risk = mb_img.remap([15, 21], [1, 1], 0)
    area_img = ee.Image.pixelArea()
    high_area = high_risk.multiply(area_img).reduceRegion(ee.Reducer.sum(), roi, 30, 1e9).getInfo()
    med_area = medium_risk.multiply(area_img).reduceRegion(ee.Reducer.sum(), roi, 30, 1e9).getInfo()
    total_area = area_img.reduceRegion(ee.Reducer.sum(), roi, 30, 1e9).getInfo()
    h = sum(high_area.values()) if high_area else 0
    m = sum(med_area.values()) if med_area else 0
    t = sum(total_area.values()) if total_area else 1
    score = (h * 1.0 + m * 0.5) / t
    return score

def classify_eri(eri):
    if eri <= 0.25:
        return 'Low Risk'
    elif eri <= 0.5:
        return 'Medium Risk'
    elif eri <= 0.75:
        return 'High Risk'
    else:
        return 'Very High Risk'

def main():
    """Função principal que orquestra todo o fluxo de trabalho."""
    setup_logging()
    
    try:
        authenticate_ee(GCP_PROJECT)
        
        # 1. DELINEAMENTO DA BACIA
        # Coordenadas corretas do exutório do Rio Jaguaribe em João Pessoa
        outlet_point = ee.Geometry.Point([-34.8256, -7.1436])
        study_area = delineate_watershed_from_outlet(outlet_point)
        
        # 2. GERAÇÃO DA SÉRIE TEMPORAL
        df = generate_monthly_timeseries(study_area, START_DATE, END_DATE)

        # 3. ERI MENSAL E EXPORTAÇÃO
        if df.empty:
            logging.warning("Nenhum dado foi retornado da análise. Verifique o período e a área de estudo.")
            return

        df = df.dropna(subset=['Chl_a', 'TSS', 'Turbidity', 'TSI'])
        if df.empty:
            logging.warning("Dados retornados não continham valores válidos para os índices.")
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results = []
        for idx, row in df.iterrows():
            # Critério 1: Water Quality
            exceed_count = int(row['Chl_a'] > 30) + int(row['Turbidity'] > 100) + int(row['TSS'] > 100)
            C_wq = exceed_count / 3.0

            # Critério 2: População
            mean_pop = get_population_density(study_area)
            C_pop = min(mean_pop / 5000.0, 1.0) if mean_pop else 0

            # Critério 3: Uso do Solo
            lu_risk_score = get_land_use_risk(study_area)

            # ERI
            ERI = 0.5 * C_wq + 0.3 * C_pop + 0.2 * lu_risk_score
            risk_class = classify_eri(ERI)

            results.append({
                'date': row['date'],
                'Chl_a': row['Chl_a'],
                'TSS': row['TSS'],
                'Turbidity': row['Turbidity'],
                'C_wq': C_wq,
                'C_pop': C_pop,
                'lu_risk_score': lu_risk_score,
                'ERI': ERI,
                'Risk_Class': risk_class
            })

        df_eri = pd.DataFrame(results)
        df_eri.to_csv(os.path.join(OUTPUT_DIR, 'environmental_risk_report_jaguaribe.csv'), index=False)
        logging.info('Relatório ERI exportado com sucesso.')
        logging.info("Workflow concluído! O próximo passo seria adicionar as funções de visualização.")
        
    except Exception as e:
        logging.error(f"Erro fatal no workflow: {e}", exc_info=True)


if __name__ == '__main__':
    main()