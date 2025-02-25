from sentinelsat import SentinelAPI
from datetime import datetime
import os
import yaml

def load_config():
    with open('configs/data_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def connect_to_copernicus(username, password):
    """Connect to Copernicus Open Access Hub"""
    api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')
    return api

def download_sentinel_data(api, area_of_interest, start_date, end_date, output_dir):
    """Download Sentinel-1 data for specified area and time period"""
    
    # Convert dates to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Query Sentinel data
    products = api.query(
        area_of_interest,
        date=(start, end),
        platformname='Sentinel-1',
        producttype='GRD'
    )
    
    # Download products
    for product_id, product_info in products.items():
        if not os.path.exists(os.path.join(output_dir, f"{product_id}.zip")):
            api.download(product_id, directory_path=output_dir)

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Connect to Copernicus
    api = connect_to_copernicus(
        config['copernicus']['username'],
        config['copernicus']['password']
    )
    
    # Download data
    download_sentinel_data(
        api,
        config['area_of_interest'],
        config['start_date'],
        config['end_date'],
        config['output_dir']
    ) 