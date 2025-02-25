from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import torch
from src.models.autoencoder import SARAutoencoder
import rasterio
import numpy as np
from pathlib import Path
import yaml
import shutil
import tempfile

app = FastAPI(
    title="SAR Water Leakage Detection API",
    description="API for detecting water leakage using SAR data"
)

# Load configuration
with open('configs/api_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model
model = SARAutoencoder(input_channels=config['model']['input_channels'])
model.load_state_dict(torch.load(config['model']['checkpoint_path']))
model.eval()

@app.post("/detect/")
async def detect_leakage(file: UploadFile = File(...)):
    """
    Detect water leakage from uploaded SAR image
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Process image
        with rasterio.open(temp_path) as src:
            image = src.read(1)
            image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
            
            # Detect anomalies
            with torch.no_grad():
                output = model(image)
                error = torch.abs(output - image)
                
                # Save results
                result_path = Path(config['api']['output_dir']) / f"result_{file.filename}"
                with rasterio.open(
                    str(result_path),
                    'w',
                    driver='GTiff',
                    height=error.shape[2],
                    width=error.shape[3],
                    count=1,
                    dtype='float32'
                ) as dst:
                    dst.write(error.numpy()[0], 1)
                
                return FileResponse(str(result_path))
    
    finally:
        # Clean up temporary file
        Path(temp_path).unlink()

@app.get("/health")
async def health_check():
    """
    Check API health
    """
    return {"status": "healthy"} 