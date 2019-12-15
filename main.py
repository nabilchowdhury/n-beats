import os
import json

from jsonschema import validate
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataloaders import *
from nbeats import *


def test_model(model, criterion, test_loader, device, plot=False):
    model.eval()
    metrics = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            metric = criterion(y, y_hat)
            metrics.append(metric.item())
    
    if plot:
        fig, ax = plt.subplots()
        ax.grid()
        plot_scatter(range(0, backcast_length), xx, color='b')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        plt.show()

    return np.average(metrics)

def train_loop_model(model, optimizer, criterion, train_loader, test_loader, epochs, device, writer=None):
    for epoch, (x, y) in enumerate(train_loader, 1):
        if epoch > epochs: break

        model.train()

        x, y = x.to(device), y.to(device)

        y_hat = model(x)
        loss = criterion(y, y_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if writer:
            writer.add_scalar('Loss/train', loss.item(), epoch)

        if epoch % 10 == 0:
            test_loss = test_model(model, criterion, test_loader, device)
            if writer:
                writer.add_scalar('Loss/test', test_loss.item(), epoch)
            print ('Epoch [{}/{}], Train Loss: {:.3f}, Test Loss: {:.3f}' 
                    .format(epoch, epochs, loss.item(), test_loss))

if __name__ == '__main__':
    # Set these parameters (later set up argparse)
    DATASET_NAME = 'M3Dataset'
    SUBSET = 'M3Year'
    L = 20
    BATCH_SIZE = 1024
    EPOCHS = 100

    DATASET = globals()[DATASET_NAME]
    FORECAST_LEN = DATASET.HORIZONS[SUBSET]
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # for m in range(2, 8):
    #     BACKCAST_LEN = m * FORECAST_LEN

    #     # Dataset and loaders
    #     train_set = DATASET(rf'data/{DATASET_NAME}/npy/{SUBSET}.npy', FORECAST_LEN, BACKCAST_LEN, MDataset.TRAIN_FOR_TEST, L=L)
    #     test_set = DATASET(rf'data/{DATASET_NAME}/npy/{SUBSET}.npy', FORECAST_LEN, BACKCAST_LEN, MDataset.TEST)
    #     train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE)
    #     test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE)

    #     # The model schema - DO NOT CHANGE THE SCHEMA UNLESS REQUIRED
    #     with open('./model_schema.json', 'r') as file:
    #         schema = json.load(file)

    #     # The model
    #     with open('./model_config.json', 'r') as file:
    #         config = json.load(file)

    #     try:
    #         validate(instance=config, schema=schema)
        
    #         model = NBEATS(FORECAST_LEN, BACKCAST_LEN, config).to(DEVICE)
    #         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
    #         # For tensorboard
    #         writer = SummaryWriter()
    #         train_loop_model(model, optimizer, NBEATSLosses.MAPE, train_loader, test_loader, EPOCHS, DEVICE, writer=writer)
    #         torch.save(model.state_dict(), f'{DATASET_NAME}_{SUBSET}_{BACKCAST_LEN}.th')

    #     except Exception as e:
    #         print(e)
    
    # Ensembling
    metrics = []
    criterion = NBEATSLosses.MAPE
    for m in range(2, 8):
        BACKCAST_LEN = m * FORECAST_LEN

        # Dataset and loaders
        test_set = DATASET(rf'data/{DATASET_NAME}/npy/{SUBSET}.npy', FORECAST_LEN, BACKCAST_LEN, MDataset.TEST)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE)
        
        # The model schema - DO NOT CHANGE THE SCHEMA UNLESS REQUIRED
        with open('./model_schema.json', 'r') as file:
            schema = json.load(file)

        # The model
        with open('./model_config.json', 'r') as file:
            config = json.load(file)

        validate(instance=config, schema=schema)
        model = NBEATS(FORECAST_LEN, BACKCAST_LEN, config).to(DEVICE)
        model.load_state_dict(torch.load(f'{DATASET_NAME}_{SUBSET}_{BACKCAST_LEN}.th'))
        
        batch_metrics = []
        batch_sizes = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_hat = model(x)
                batch_metric = criterion(y, y_hat)
                batch_metrics.append(batch_metric.item())
                batch_sizes.append(x.shape[0])

        metrics.append(np.average(batch_metrics, weights=batch_sizes))
    
    print(np.average(metrics))

    # with torch.no_grad():
    #     ls, ws = [], []
    #     for horizon in DATASET.HORIZONS:
    #         FORECAST_LEN = DATASET.HORIZONS[horizon]
    #         test_set = DATASET(rf'data/{DATASET_NAME}/npy/{horizon}.npy', FORECAST_LEN, BACKCAST_LEN, MDataset.TEST)
    #         test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE)
    #         losses = []
    #         weights = []
    #         w = 0
    #         for x, y in test_loader:
    #             y_naive = x[:, -1].view(-1, 1).repeat(1, FORECAST_LEN) 
    #             loss = NBEATSLosses.SMAPE(y, y_naive)
    #             losses.append(loss)
    #             weights.append(y.shape[0])
    #             w += y.shape[0]
    #         ls.append(np.average(losses, weights=weights))
    #         ws.append(w)

    #     print(np.average(ls, weights=ws))
        
        # losses = []
        # for horizon in DATASET.HORIZONS:
        #     FORECAST_LEN = DATASET.HORIZONS[horizon]
        #     test_set = DATASET(rf'data/{DATASET_NAME}/npy/{horizon}.npy', FORECAST_LEN, BACKCAST_LEN, MDataset.TEST)
        #     test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE)
        #     for x, y in test_loader:
        #         y_naive = x[:, -1].view(-1, 1).repeat(1, FORECAST_LEN) 
                
        #         for i in range(y.shape[0]):
        #             s = 0
        #             for j in range(y.shape[1]):
        #                 s += abs(y[i][j] - y_naive[i][j]) / (abs(y[i][j]) + abs(y_naive[i][j]))
        #             s = s * 200 / FORECAST_LEN
        #             losses.append(s)


        # print(np.average(losses))
