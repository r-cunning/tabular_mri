
import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

class TabularAutoencoderDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataframe):
        """
        Args:
            file_path (string): Path to the dataset file.
            target_column (string): Name of the target column.
            feature_columns (list, optional): List of column names you want to use as features.
                                             If None, all columns except the target will be used.
        """
        self.dataframe = dataframe 
        self.X = self.dataframe
        self.y = self.X

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def to_absolute(x):
        return np.abs(x)


class AutoEncoderDataModule(pl.LightningDataModule):
    def __init__(self, dataframe, batch_size, num_workers, sample_id_col, scale=True, test_size= 0.4, scaler_func=StandardScaler, split_data=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.sample_id_col = sample_id_col
        self.dataframe = dataframe
        self.test_size = test_size 
        
        self.scale = scale
        self.split_data = split_data
        self.scaler_func = scaler_func
        
        self.prepare_data()
        
        
    def prepare_data(self):
        
        if self.split_data:
            self.train_val_test_split()
        
            if self.scale==True:
                scaler = self.scaler_func()

                self.X_train = scaler.fit_transform(self.X_train)
                self.X_val = scaler.transform(self.X_val)
                self.X_test = scaler.transform(self.X_test)

            self.train_ds = TabularAutoencoderDataset(self.X_train)
            self.val_ds = TabularAutoencoderDataset(self.X_val)
            self.test_ds = TabularAutoencoderDataset(self.X_test)
                
                
        else:
            self.X_train = self.dataframe

            if self.scale==True:
                scaler = self.scaler_func()
                self.X_train = scaler.fit_transform(self.X_train)

            self.train_ds = self.val_ds = self.test_ds = TabularAutoencoderDataset(self.X_train)
        
        
    def train_val_test_split(self, random_state=42):
        
        self.X_train, X_temp = train_test_split(self.dataframe, test_size=self.test_size, random_state=random_state)
        self.X_val, self.X_test = train_test_split(X_temp, test_size=0.5, random_state=random_state)

        self.train_ids = self.X_train[self.sample_id_col]
        self.X_train = self.X_train.drop(columns=[self.sample_id_col]).values
        
        self.val_ids = self.X_val[self.sample_id_col]
        self.X_val = self.X_val.drop(columns=[self.sample_id_col]).values
        self.test_ids = self.X_test[self.sample_id_col]
        self.X_test = self.X_test.drop(columns=[self.sample_id_col]).values
        
                
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False)
        
    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False)
