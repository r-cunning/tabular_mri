import pandas as pd
import numpy as np
import os
from src.utils.etl.dataframe_tools import column_filter
from collections import OrderedDict




class BasePandasDataset():
    
    def __init__(self, dataframe, target, sample_id_col = None, feature_filter: dict = None, 
                 match_all: bool = True, dataset_name = 'tabular_dataset'):
        """
        Args:
            file_path (string): Path to the dataset file.
            target_column (string): Name of the target column.
            feature_columns (list, optional): List of column names you want to use as features.
                                             If None, all columns except the target will be used.
        """
        self.dataframe = dataframe.copy()
        self.set_target(target)
        self.set_name(dataset_name)
        
        if feature_filter:
            self.X = column_filter(self.dataframe, feature_filter, match_all)
            self.feature_names = self.X.columns
            self.sample_ids = self.dataframe[sample_id_col] if sample_id_col else self.dataframe.index
            # self.features = self.dataframe[feature_columns].values
        else:
            drop_cols = [target, sample_id_col] if sample_id_col else [target]
            self.X = self.dataframe.drop(columns=drop_cols).astype(np.float32)
            self.feature_names = self.X.columns
            self.sample_ids = self.dataframe[sample_id_col] if sample_id_col else self.dataframe.index





    def __len__(self):
        return len(self.dataframe)
    
    
    def get_features(self):
        return self.feature_names
    
    def set_name(self, name):
        self.name = name
    
    def set_target(self, target_column: str):
        try:
            self.y = self.dataframe[target_column]
            self.target_name = target_column
        except KeyError:
            print("Target column not found. Please double check your column names!")

    def set_inputs(self, features: list):
        self.X = self.dataframe[features]
        self.feature_names = features
        return self
    
    
    def getID(self, idx):
        return self.sample_ids[idx]
    
    



class RandomPandasDataset(BasePandasDataset):
    
    def __init__(self, pandas_dataset: BasePandasDataset):
        self.rows, self.cols = pandas_dataset.X.shape
        self.random_array = np.random.rand(self.rows, self.cols)
        
        self.sample_ids = pandas_dataset.sample_ids
        self.feature_names = pandas_dataset.feature_names
        self.X = pd.DataFrame(self.random_array, columns=self.feature_names)
        self.y = pandas_dataset.y
        
        self.dataframe = pd.concat([self.X, self.y], axis=1)

        
        
        
    
