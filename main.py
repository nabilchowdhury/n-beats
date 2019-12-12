import os
import json

from jsonschema import validate
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataloaders import *
from nbeats import *


def test_model(model, criterion, test_loader):
    model.eval()
    metrics = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = model(x)
            metric = criterion(y, y_hat, FORECAST_LEN)
            metrics.append(metric.item())
    
    return np.average(metrics)

def train_loop_model(model, optimizer, criterion, train_loader, test_loader, epochs, writer=None):
    model.train()
    for epoch, (x, y) in enumerate(train_loader, 1):
        if epoch > epochs: break
        x, y = x.to(DEVICE), y.to(DEVICE)

        y_hat = model(x)
        loss = criterion(y, y_hat, FORECAST_LEN)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if writer:
            writer.add_scalar('Loss/train', loss.item(), epoch)

        if epoch % 10 == 0:
            test_loss = test_model(model, criterion, test_loader)
            if writer:
                writer.add_scalar('Loss/test', test_loss.item(), epoch)
            print ('Epoch [{}/{}], Train Loss: {:.3f}, Test Loss: {:.3f}' 
                    .format(epoch, epochs, loss.item(), test_loss))

if __name__ == '__main__':
    # For tensorboard
    writer = SummaryWriter()

    # Hyper parameters
    FORECAST_LEN = 6
    BACKCAST_LEN = 4 * FORECAST_LEN
    BATCH_SIZE = 1024
    EPOCHS = 5000
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    # Dataset and loaders
    train_set = M4Dataset('data/M4/npy/Yearly.npy', FORECAST_LEN, BACKCAST_LEN, MDataset.TRAIN_FOR_TEST, L=1.5*FORECAST_LEN)
    test_set = M4Dataset('data/M4/npy/Yearly.npy', FORECAST_LEN, BACKCAST_LEN, MDataset.TEST)
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
    
        model = NBEATS(FORECAST_LEN, BACKCAST_LEN, config).to(DEVICE)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # train_loop_model(model, optimizer, NBEATSLosses.MAPE, train_loader, test_loader, EPOCHS, writer=writer)
        # torch.save(model.state_dict(), os.path.join(f'./model_yearly_{BACKCAST_LEN}.model'))
    except Exception as e:
        print(e)
