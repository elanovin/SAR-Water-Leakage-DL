import torch
from src.models.autoencoder import SARAutoencoder
import rasterio
import numpy as np
from pathlib import Path
import yaml
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def detect_anomalies(model, image, threshold):
    """Detect anomalies using reconstruction error"""
    model.eval()
    with torch.no_grad():
        # Forward pass
        output = model(image)
        
        # Calculate reconstruction error
        error = torch.abs(output - image)
        
        # Threshold to identify anomalies
        anomalies = error > threshold
        
        return anomalies, error

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SARAutoencoder(input_channels=config['model']['input_channels'])
    model.load_state_dict(torch.load(config['model']['checkpoint_path']))
    model = model.to(device)
    
    # Process input data
    input_dir = Path(config['inference']['input_dir'])
    output_dir = Path(config['inference']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    for file in input_dir.glob('*.tiff'):
        # Load and preprocess image
        with rasterio.open(str(file)) as src:
            image = src.read(1)
            image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
            image = image.to(device)
            
            # Detect anomalies
            anomalies, error = detect_anomalies(
                model,
                image,
                config['inference']['threshold']
            )
            
            # Save results
            output_path = output_dir / f"anomalies_{file.name}"
            with rasterio.open(
                str(output_path),
                'w',
                driver='GTiff',
                height=error.shape[2],
                width=error.shape[3],
                count=1,
                dtype='float32'
            ) as dst:
                dst.write(error.cpu().numpy()[0], 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/inference_config.yaml')
    args = parser.parse_args()
    
    main(args.config) 