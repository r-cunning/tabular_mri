import pandas as pd
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class TabularDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, y):
        """
        Args:
            file_path (string): Path to the dataset file.
            target_column (string): Name of the target column.
            feature_columns (list, optional): List of column names you want to use as features.
                                             If None, all columns except the target will be used.
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    

class TabularDataModuleLOOCV(pl.LightningDataModule):
    def __init__(self, X, y, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers


        self.train_ds = TabularDataset(X, y)
        

        
                
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True)
