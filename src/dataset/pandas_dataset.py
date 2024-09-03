import pandas as pd
import numpy as np
import os
from src.utils.etl.dataframe_tools import column_filter
from collections import OrderedDict
from sklearn.model_selection import train_test_split



class BasePandasDataset():
    
    def __init__(self, dataframe, target, sample_id_col = None, feature_filter: dict = None, 
                 match_all: bool = True, dataset_name = 'tabular_dataset', train_test_split=False, test_size=0.4):
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
        self.test_size = test_size
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

        if train_test_split ==True:
            self.train_val_test_split()

    def train_val_test_split(self, random_state=None):
        
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=self.test_size, random_state=42)


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
        self.feature_names = pandas_dataset.X.columns
        self.X = pd.DataFrame(self.random_array, columns=self.feature_names)
        self.y = pandas_dataset.y
        
        self.dataframe = pd.concat([self.X, self.y], axis=1)

        
        
        
    
