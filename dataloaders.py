import os
import sys

import numpy as np
import pandas as pd
import torch

class MDataset(torch.utils.data.Dataset):
    TRAIN_FOR_VALIDATION = 0
    TRAIN_FOR_TEST = 1
    VALIDATION = 2
    TEST = 3
    
    def __init__(self, path, horizon, lookback, dataset_type, L=None):
        self.horizon = horizon
        self.lookback = lookback
        self.dataset_type = dataset_type
        self.L = L
        
        self.data = torch.from_numpy(np.load(path)).float()
        self.data_len = self.data.shape[0]
        self.ts_len = self.data.shape[1]
        
    def __getitem__(self, idx):
        n_y = None # The number forecast samples during training. Since we pick a random point, it may be less than the horizon length.
        
        if self.dataset_type == MDataset.TRAIN_FOR_VALIDATION:
            # Ignore the actual index during training, we set it to a random idx in the dataset
            idx = np.random.randint(self.data_len)
            anchor_idx = np.random.randint(2 * self.horizon + 1, 2 * self.horizon + self.L + 1)
            n_y = min(anchor_idx - 2 * self.horizon, self.horizon)
            
        elif self.dataset_type == MDataset.TRAIN_FOR_TEST:
            # Ignore the actual index during training, we set it to a random idx in the dataset
            idx = np.random.randint(self.data_len)
            # anchor_idx = np.random.randint(self.horizon + 1, self.horizon + self.L + 1)
            # n_y = min(anchor_idx - self.horizon, self.horizon)
            anchor_idx =  2 * self.horizon

        elif self.dataset_type == MDataset.VALIDATION:
            anchor_idx = 2 * self.horizon
            
        elif self.dataset_type == MDataset.TEST:
            anchor_idx = self.horizon
        
        # Make sure to clone here so we don't modify original data source
        x = self.data[idx, self.ts_len - anchor_idx - self.lookback : self.ts_len - anchor_idx].clone()
        y = self.data[idx, self.ts_len - anchor_idx : self.ts_len - anchor_idx + self.horizon].clone()
        
        if (x != x).all():
            print('x')
            embed()
        if (y != y).all():
            print('y')
            embed()

        x[x != x] = 0
        y[y != y] = 0
        
        if x.shape[0] < self.lookback:
            x = torch.cat((torch.zeros(self.lookback - x.shape[0]), x))
        
        # If training, we need to zero out samples after the first n_y samples because they seep into the validation or test sets. 
        if n_y:
            y[(self.horizon - n_y):] = 0
        
        return x, y
    
    def __len__(self):
        if self.dataset_type == MDataset.VALIDATION or self.dataset_type == MDataset.TEST:
            return self.data_len
        else:
            return sys.maxsize # Infinite length for training, we control how many loops to do


class M4Dataset(MDataset):
    """A PyTorch dataset for M4.
    """
    
    HORIZONS = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    
    @staticmethod
    def convert_csv_to_npy(train_dir, test_dir, output_dir):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            
        # For each set of train and test csvs
        for horizon in M4Dataset.HORIZONS:
            # Contains historical time series data (per row). Series may be of different lengths, so shorter ones are nan filled by default.
            data = pd.read_csv(os.path.join(train_dir, horizon + '-train.csv'), index_col=0).values
            # Contains the forecast for each time series in data. All forecasts for a horizon are of the same forecast length.
            test_data = pd.read_csv(os.path.join(test_dir, horizon + '-test.csv'), index_col=0).values
            
            # Roll time series (each row) to move nans to beginning.
            for i, ts in enumerate(data):
                # Retrieve indices of all NaNs, but we only care about the first occurrence.
                nan_idx = np.argwhere(np.isnan(ts))
                if nan_idx.size > 0:
                    # Shift NaNs to beginning of time series instead of the end
                    data[i] = np.roll(ts, data.shape[1] - nan_idx[0])
            
            # Merge forecasts with historical data for easier data manipulation during training
            data = np.hstack((data, test_data))
            np.save(os.path.join(output_dir, horizon + '.npy'), data)
    
    def __init__(self, path, horizon, lookback, dataset_type, L=None):
        super().__init__(path, horizon, lookback, dataset_type, L)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
    
    def __len__(self):
        return super().__len__()


class M3Dataset(MDataset):
    
    HORIZONS = ['M3Year', 'M3Quart', 'M3Month', 'M3Other']
    
    @staticmethod
    def convert_xls_to_npy(path, output_dir):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            
        xls = pd.ExcelFile(path)
        
        # For each set of train and test csvs
        for horizon in M3Dataset.HORIZONS:
            # Contains historical time series data (per row). Series may be of different lengths, so shorter ones are nan filled by default.
            data = xls.parse(horizon).iloc[:, 6:].values
            
            # Roll time series (each row) to move nans to beginning.
            for i, ts in enumerate(data):
                # Retrieve indices of all NaNs, but we only care about the first occurrence.
                nan_idx = np.argwhere(np.isnan(ts))
                if nan_idx.size > 0:
                    # Shift NaNs to beginning of time series instead of the end
                    data[i] = np.roll(ts, data.shape[1] - nan_idx[0])
            
            np.save(os.path.join(output_dir, horizon + '.npy'), data)
    
    def __init__(self, path, horizon, lookback, dataset_type, L=None):
        super().__init__(path, horizon, lookback, dataset_type, L)
        
    def __getitem__(self, idx):
        return super().__getitem__(idx)
        
    def __len__(self):
        return super().__len__()

if __name__ == '__main__':
    M3Dataset.convert_xls_to_npy(r'data/M3/M3C.xls', output_dir=r'data/M3/npy')
    M4Dataset.convert_csv_to_npy(train_dir=r'data/M4/Train',
                                 test_dir=r'data/M4/Test',
                                 output_dir=r'data/M4/npy')