import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.autoencoder import SARAutoencoder
from src.data.dataset import SARDataset
import yaml
import logging
from pathlib import Path
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(model, train_loader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SARAutoencoder(input_channels=config['model']['input_channels'])
    model = model.to(device)
    
    # Setup dataset and dataloader
    dataset = SARDataset(config['data']['processed_dir'])
    train_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    criterion = nn.MSELoss()
    
    # Train model
    train_model(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        config['training']['epochs']
    )
    
    # Save model
    model_path = Path(config['model']['save_dir']) / 'model.pth'
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/autoencoder_config.yaml')
    args = parser.parse_args()
    
    main(args.config) 