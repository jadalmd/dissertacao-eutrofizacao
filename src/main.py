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

        # 3. PÓS-PROCESSAMENTO E EXPORTAÇÃO
        if df.empty:
            logging.warning("Nenhum dado foi retornado da análise. Verifique o período e a área de estudo.")
            return
            
        df = df.dropna(subset=['Chl_a', 'TSS', 'Turbidity', 'TSI'])
        if df.empty:
            logging.warning("Dados retornados não continham valores válidos para os índices.")
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DIR, f'analise_jaguaribe_{START_DATE}_a_{END_DATE}.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Resultados exportados para: {csv_path}")

        # 4. VISUALIZAÇÕES (Placeholder - podem ser adicionadas aqui depois)
        logging.info("Workflow concluído! O próximo passo seria adicionar as funções de visualização.")
        
    except Exception as e:
        logging.error(f"Erro fatal no workflow: {e}", exc_info=True)


if __name__ == '__main__':
    main()