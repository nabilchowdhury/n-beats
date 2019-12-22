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
        
        self.data = np.load(path)
        self.data_len, self.ts_len = self.data.shape
        
    def __getitem__(self, idx):
        if self.dataset_type in (MDataset.TRAIN_FOR_VALIDATION, MDataset.TRAIN_FOR_TEST):
            idx = np.random.randint(self.data_len)
        
        ts = self.data[idx].copy() # So as to not accidentally change the actual data
        nan_indices = np.argwhere(np.isnan(ts)).ravel()
        start_idx = nan_indices[-1] + 1 if nan_indices.shape[0] else 0

        if self.dataset_type == MDataset.TRAIN_FOR_VALIDATION:
            L = np.minimum(self.L, self.ts_len - 2 * self.horizon)
            anchor_idx = np.random.randint(2 * self.horizon + 1, 2 * self.horizon + L + 1)
            x_start_idx = min(anchor_idx + self.lookback, self.ts_len - start_idx)
            y_end_idx = max(anchor_idx - self.horizon, 2 * self.horizon)

        elif self.dataset_type == MDataset.TRAIN_FOR_TEST:
            L = np.minimum(self.L, self.ts_len - self.horizon - start_idx - 1)
            anchor_idx = np.random.randint(self.horizon + 1, self.horizon + L + 1)
            x_start_idx = min(anchor_idx + self.lookback, self.ts_len - start_idx)
            y_end_idx = max(anchor_idx - self.horizon, self.horizon)

        elif self.dataset_type == MDataset.VALIDATION:
            anchor_idx = 2 * self.horizon
            x_start_idx = min(anchor_idx + self.lookback, self.ts_len - start_idx)
            y_end_idx = self.horizon

        elif self.dataset_type == MDataset.TEST:
            anchor_idx = self.horizon
            x_start_idx = min(anchor_idx + self.lookback, self.ts_len - start_idx)
            y_end_idx = 0

        x = np.append( np.zeros(self.lookback + anchor_idx - x_start_idx), ts[-x_start_idx : -anchor_idx] )
        y = np.append( ts[-anchor_idx : -y_end_idx] if y_end_idx else ts[-anchor_idx:] , np.zeros(self.horizon - anchor_idx + y_end_idx))

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    
    def __len__(self):
        if self.dataset_type == MDataset.VALIDATION or self.dataset_type == MDataset.TEST:
            return self.data_len
        else:
            return sys.maxsize # Infinite length for training, we control how many loops to do


class M4Dataset(MDataset):
    """A PyTorch dataset for M4.
    """
    
    HORIZONS = dict(Yearly=6, Quarterly=8, Monthly=18, Weekly=13, Daily=14, Hourly=48)
    L = dict(Yearly=9, Quarterly=12, Monthly=27, Weekly=130, Daily=140, Hourly=480)
    
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
    
    HORIZONS = dict(M3Year=6, M3Quart=8, M3Month=18, M3Other=8)
    L = dict(M3Year=20, M3Quart=20, M3Month=20, M3Other=10)
    
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


class DummyDataset(MDataset):
    @staticmethod
    def generate_dummy_data(output_dir, number_of_ts=2048, length_of_ts=100):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        linspace = np.linspace(0, 30, length_of_ts)
        matrix = linspace.copy()
        for _ in range(number_of_ts - 1):
            matrix = np.vstack((matrix, linspace))

        data = np.cos(2 * np.random.randint(low=1, high=3, size=(nu/mber_of_ts, length_of_ts)) * np.pi * matrix)
        data += np.cos(2 * np.random.randint(low=2, high=4, size=(number_of_ts, length_of_ts)) * np.pi * matrix)
        data += matrix + np.random.rand(number_of_ts, length_of_ts) * 0.1

        np.save(os.path.join(output_dir, 'dummy.npy'), data)
        
    def __init__(self, path, horizon, lookback, dataset_type, L=None):
        super().__init__(path, horizon, lookback, dataset_type, L)
        
    def __getitem__(self, idx):
        return super().__getitem__(idx)
        
    def __len__(self):
        return super().__len__()

if __name__ == '__main__':
    M3Dataset.convert_xls_to_npy(r'data/M3/M3C.xls', output_dir=r'data/M3/npy')
    M4Dataset.convert_csv_to_npy(train_dir=r'data/M4/Train', test_dir=r'data/M4/Test', output_dir=r'data/M4/npy')
    DummyDataset.generate_dummy_data(r'data/Dummy/npy')
    