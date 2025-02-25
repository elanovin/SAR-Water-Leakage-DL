# Deep Learning SAR Water Leakage Detection

This project implements an unsupervised deep learning approach for detecting underground water leakage using Sentinel-1 SAR (Synthetic Aperture Radar) data. The system analyzes surface deformations and moisture patterns to identify potential leakage locations without requiring labeled training data.

## Features
- Automated processing of Sentinel-1 SAR data
- Unsupervised anomaly detection using deep autoencoders
- Surface deformation analysis using InSAR processing
- Soil moisture estimation from backscatter coefficients
- Interactive visualization tools for detected anomalies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Deep-Learning-SAR-Water-Leakage.git
cd Deep-Learning-SAR-Water-Leakage
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
Deep-Learning-SAR-Water-Leakage/
│── data/                    # SAR data and processed features
│── notebooks/               # Jupyter notebooks for analysis
│── src/                     # Model training and inference scripts
│── models/                  # Saved trained models
│── api/                     # FastAPI deployment code
│── visualization/           # GIS & heatmap visualization tools
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
│── train.py                # Training script
│── inference.py            # Model inference script
```

## Usage

1. Data Preparation:
```bash
python src/data/download_sentinel.py --start_date YYYY-MM-DD --end_date YYYY-MM-DD
python src/data/preprocess.py
```

2. Model Training:
```bash
python train.py --config configs/autoencoder_config.yaml
```

3. Inference:
```bash
python inference.py --input_data path/to/sar/data --model_path path/to/model
```

4. API Service:
```bash
uvicorn api.main:app --reload
```

## Contributing
Elaheh Novinfard at elanvnfrd@gmail.com.


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- European Space Agency (ESA) for providing Sentinel-1 data
- Copernicus Open Access Hub for data access 