import argparse
import json
import os

from jsonschema import validate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataloaders import *
from nbeats import *


def test_model(model, criterion, test_loader, device, use_mask):
    metrics = []
    weights = []
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x_mask = (x != 0) if use_mask else None
            y_mask = (y != 0) if use_mask else None
            y_hat = model(x, x_mask, y_mask)
            metric = criterion(y, y_hat)
            metrics.append(metric.item())
            weights.append(x.shape[0])

    return np.average(metrics, weights=weights)

def train_loop_model(model, optimizer, criterion, train_loader, test_loader, epochs, device, use_mask, progress_freq, writer=None):
    for epoch, (x, y) in enumerate(train_loader, 1):
        if epoch > epochs: break

        model.train()

        x, y = x.to(device), y.to(device)
        x_mask = (x != 0) if use_mask else None
        y_mask = (y != 0) if use_mask else None
        y_hat = model(x, x_mask, y_mask)
        loss = criterion(y, y_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if writer:
            writer.add_scalar('Loss/train', loss.item(), epoch)

        if epoch % progress_freq == 0:
            test_loss = test_model(model, criterion, test_loader, device, use_mask)
            if writer:
                writer.add_scalar('Loss/test', test_loss.item(), epoch)
            print ('Epoch [{}/{}], Train Loss: {:.3f}, Test Loss: {:.3f}'.format(epoch, epochs, loss.item(), test_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N-Beats Model')
    parser.add_argument('--dataset-name', '-ds', help='Name of Dataset eg. M3Dataset', type=str, required=True)
    parser.add_argument('--horizons', '-hz', help='Horizon for dataset eg. M3Year', nargs='+', type=str, required=True)
    parser.add_argument('--batch-size', '-bs', help='Batch size', type=int, required=True)
    parser.add_argument('--epochs', '-e', help='Epochs', type=int, required=True)
    parser.add_argument('--lookback-multipliers', '-lb', help='Size of the input is multiplier * length of horizon', nargs='+', type=int, required=True)
    parser.add_argument('--metrics', '-m', help='One or more of smape, mape, mase', nargs='+', type=str, required=True)
    parser.add_argument('--set-seed', '-seed', help='Set the random seeds for reproducibility', action='store_true')
    parser.add_argument('--no-mask', '-no-mask', help='If true, does not use masking during training', action='store_true')
    parser.add_argument('--use-tensorboard', '-tboard', help='Output to tensorboard (./runs)', action='store_true')
    parser.add_argument('--progress-freq', '-pf', help='Display train and test loss every pf epochs', type=int, default=10)
    parser.add_argument('--output_dir', '-out', help='Output directory for trained models', type=str, default='./models')
    args = parser.parse_args()

    if args.set_seed:
        torch.manual_seed(0)
        np.random.seed(0)
    
    DATASET_NAME = args.dataset_name
    HORIZONS = args.horizons
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LOOKBACK_MULTIPLIERS = args.lookback_multipliers
    METRIC_NAMES = [metric.upper() for metric in args.metrics]
    USE_MASK = not args.no_mask
    USE_TENSORBOARD = args.use_tensorboard
    PROGRESS_FREQ = args.progress_freq
    OUTPUT_DIR = args.output_dir

    DATASET = globals()[DATASET_NAME]
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    for METRIC_NAME in METRIC_NAMES:
        # Get the loss function by name
        METRIC = getattr(globals()['NBEATSLosses'], METRIC_NAME)

        for HORIZON in HORIZONS:
            FORECAST_LEN = DATASET.HORIZONS[HORIZON]
            L = DATASET.L[HORIZON]

            for M in LOOKBACK_MULTIPLIERS:
                BACKCAST_LEN = M * FORECAST_LEN
                print(f'Metric: {METRIC_NAME}, Horizon: {HORIZON}, Backcast length: {BACKCAST_LEN}')

                # Dataset and loaders
                train_set = DATASET(rf'data/{DATASET_NAME}/npy/{HORIZON}.npy', FORECAST_LEN, BACKCAST_LEN, MDataset.TRAIN_FOR_TEST, L=L)
                test_set = DATASET(rf'data/{DATASET_NAME}/npy/{HORIZON}.npy', FORECAST_LEN, BACKCAST_LEN, MDataset.TEST)
                train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE)
                test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE)

                # The model schema - DO NOT CHANGE THE SCHEMA UNLESS REQUIRED
                with open('./model_schema.json', 'r') as file:
                    schema = json.load(file)

                # The model
                with open('./model_config.json', 'r') as file:
                    config = json.load(file)

                try:
                    validate(instance=config, schema=schema)
                except Exception as e:
                    print(e)
                
                model = NBEATS(FORECAST_LEN, BACKCAST_LEN, config).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                
                # For tensorboard
                writer = SummaryWriter() if USE_TENSORBOARD else None
                train_loop_model(model, optimizer, METRIC, train_loader, test_loader, EPOCHS, DEVICE, USE_MASK, PROGRESS_FREQ, writer=writer)

                # Save model
                if not os.path.isdir(OUTPUT_DIR):
                    os.mkdir(OUTPUT_DIR)
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_{HORIZON}_{BACKCAST_LEN}_{METRIC_NAME}.th'))

