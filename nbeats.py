import torch
import torch.nn as nn
from torch.nn import functional as F

from IPython import embed

# Mask inputs to stacks and block and output (loss)

class Block(nn.Module):
    def __init__(self,
                 forecast_len,
                 backcast_len,
                 theta_forecast_dim,
                 theta_backcast_dim,
                 hidden_dims=[256, 512, 512, 2048]):
        super().__init__()
        
        # Pass input x_l through the hidden layers (the paper has 4 layers but we allow arbitrary).
        fc_dims = [backcast_len] + hidden_dims
        self.fcs = nn.ModuleList([nn.Linear(fc_dims[i], fc_dims[i + 1]) for i in range(len(fc_dims) - 1)])
        
        # Pass the last hidden layer to the theta_forecast and theta_backcast layers.
        self.fc_theta_forecast = nn.Linear(fc_dims[-1], theta_forecast_dim)
        self.fc_theta_backcast = nn.Linear(fc_dims[-1], theta_backcast_dim)
        
        # Pass the fc_theta layers to the respective basis function projection layers
        self.fc_forecast_basis = nn.Linear(theta_forecast_dim, forecast_len)
        self.fc_backcast_basis = nn.Linear(theta_backcast_dim, backcast_len)
    
    def forward(self, x):
        # Pass x through the fc layers
        for fc in self.fcs:
            x = F.relu(fc(x))
        
        # Take the last layers output and pass to these layers to get theta_forecast and theta_backcast
        theta_forecast = self.fc_theta_forecast(x)
        theta_backcast = self.fc_theta_backcast(x)
        
        # Feed thetas into basis layer to obtain forecast and backcast
        forecast = self.fc_forecast_basis(theta_forecast)
        backcast = self.fc_backcast_basis(theta_backcast)
        
        return forecast, backcast

class Stack(nn.Module):
    def __init__(self,
                 forecast_len,
                 backcast_len,
                 stack_config):
        super().__init__()
        
        self.forecast_len = forecast_len
        block_configs = stack_config["blocks"]

        self.blocks = nn.ModuleList([])
        for block_config in block_configs:
            multiply = block_config.get('multiply', 1)
            self.blocks.extend((Block(forecast_len, backcast_len, block_config["theta_forecast_dim"], block_config["theta_backcast_dim"], block_config["hidden_dims"]) for _ in range(multiply)))

    def forward(self, x):
        forecast = torch.zeros((x.shape[0], self.forecast_len), device=x.device)
        for block in self.blocks:
            f, b = block(x)
            forecast = forecast + f # += is an inplace operation which affects autograd, so we do forecast = forecast + f instead
            x = x - b
        
        return forecast, x # backcast and x are the same thing
    
class NBEATS(nn.Module):
    def __init__(self, forecast_len, backcast_len, config):
        super().__init__()
        
        self.forecast_len = forecast_len
        backcast_len = backcast_len
        stack_configs = config["stacks"]
        
        self.stacks = nn.ModuleList([])
        for stack_config in stack_configs:
            multiply = stack_config.get('multiply', 1)
            self.stacks.extend(Stack(forecast_len, backcast_len, stack_config) for _ in range(multiply))

        
    def forward(self, x):
        forecast = torch.zeros((x.shape[0], self.forecast_len), device=x.device)
        for stack in self.stacks:
            f, x = stack(x)
            forecast = forecast + f
        
        return forecast


class NBEATSLosses:
    @staticmethod
    def SMAPE(y, y_hat): # Add stop gradient and diff no nan
        horizon = y.shape[1]
        mask = (y != 0)
        y = torch.masked_select(y, mask)
        y_hat = torch.masked_select(y_hat, mask)
        loss = 200 * torch.mean(torch.abs(y - y_hat) / (torch.abs(y) + torch.abs(y_hat)))
        return loss
    
    @staticmethod
    def MAPE(y, y_hat): # Add diff no nan
        horizon = y.shape[1]
        mask = (y != 0)
        y = torch.masked_select(y, mask)
        y_hat = torch.masked_select(y_hat, mask)
        loss = 100 * torch.mean(torch.abs(y - y_hat) / torch.abs(y))
        return loss
    
    @staticmethod
    def MASE(y, y_hat):
        raise NotImplementedError