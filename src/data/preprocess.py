import rasterio
import numpy as np
from pathlib import Path
import os
from osgeo import gdal
import yaml

def load_config():
    with open('configs/preprocessing_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def apply_speckle_filter(image):
    """Apply Lee speckle filter to SAR image"""
    from scipy.ndimage import uniform_filter
    from scipy.ndimage import gaussian_filter
    
    # Calculate local statistics
    mean = uniform_filter(image, size=5)
    mean_square = uniform_filter(image**2, size=5)
    variance = mean_square - mean**2
    
    # Calculate weights
    noise_variance = variance.mean()
    weights = variance / (variance + noise_variance)
    
    # Apply filter
    filtered = mean + weights * (image - mean)
    return filtered

def calculate_backscatter(image):
    """Convert DN values to backscatter coefficients"""
    # This is a simplified version - actual conversion depends on specific SAR product
    return 10 * np.log10(image)

def process_sar_image(input_path, output_path):
    """Process a single SAR image"""
    with rasterio.open(input_path) as src:
        image = src.read(1)
        
        # Apply preprocessing steps
        filtered = apply_speckle_filter(image)
        backscatter = calculate_backscatter(filtered)
        
        # Normalize data
        normalized = (backscatter - backscatter.mean()) / backscatter.std()
        
        # Save processed image
        profile = src.profile
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(normalized, 1)

def main():
    config = load_config()
    
    input_dir = Path(config['input_dir'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Process all SAR images in input directory
    for file in input_dir.glob('*.tiff'):
        output_path = output_dir / f"processed_{file.name}"
        process_sar_image(str(file), str(output_path))

if __name__ == "__main__":
    main() 