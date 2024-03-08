import torch.optim as optim
from torchmetrics import MeanAbsoluteError
import torch.nn.functional as F
from torchmetrics.metric import Metric
from torch.optim import Optimizer

import lightning as pl
from lightning.fabric.loggers import TensorBoardLogger


from src.lightning.training import kfold
from src.lightning.evaluation.shapley import ShapDeepExplainer as shapley
from src.lightning.models.mlp_regressor import SimpleMLP


from src.dataset.pandas_dataset import BasePandasDataset as bpd
from src.project_system import project


class Hyperparameters:
    def __init__(self, base_model: pl.LightningModule = SimpleMLP, hidden_size:int = 32, dataset: bpd = None, 
                 batch_size: int = 3, epochs: int = 60, lr: float = 0.001, 
                 training_regime = kfold, data_transforms: dict = None,
                 dry_run: bool = False, seed: int = 42,
                 log_interval: int = 2, optimizer: Optimizer = optim.Adam,
                 loss_function = F.l1_loss, acc_metric: Metric = MeanAbsoluteError,
                 feature_eval = shapley, folds: int = 24, loocv: bool = True,
                 
                 save_model: bool = False, logger = TensorBoardLogger, 
                 project_name = "default_project", experiment_name = "default_experiment", run_name = "default_run"):
        
        
        self.base_model = base_model
        self.input_size = dataset.X.shape[1]
        self.hidden_size = hidden_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.training_regime = training_regime
        self.data_transforms = data_transforms
        self.dry_run = dry_run
        self.seed = seed
        self.log_interval = log_interval
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.acc_metric = acc_metric
        self.feature_eval = feature_eval
        self.folds = folds
        self.loocv = loocv
        self.save_model = save_model
        self.logger = logger
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.run_name = run_name
        



import matplotlib.pyplot as plt
import shap


class TorchRun:
    def __init__(self, base_model: pl.LightningModule = SimpleMLP, hidden_size:int = 32, dataset: bpd = None, 
                 batch_size: int = 3, epochs: int = 60, 
                 lr: float = 0.001, training_regime = kfold, data_transforms: dict = None,
                 dry_run: bool = False, seed: int = 42,
                 log_interval: int = 2, optimizer: Optimizer = optim.Adam,
                 loss_function = F.l1_loss,
                 acc_metric: Metric = MeanAbsoluteError,
                 feature_eval = shapley,
                 folds: int = 24, loocv: bool = True,
                 save_model: bool = False,
                 logger = TensorBoardLogger, 
                 project_name = "default", experiment_name = "default", run_name = "default"
                 ):
        
        
        
        # Create Hyperparameters object
        self.hparams = Hyperparameters(base_model, hidden_size, dataset, batch_size, epochs, lr, 
                                       training_regime, data_transforms,
                                       dry_run, seed, 
                                       log_interval, optimizer, 
                                       loss_function, acc_metric,
                                       feature_eval, folds, loocv, 
                                       save_model, logger, 
                                       project_name, experiment_name, run_name)
        
        self.reports = "No report yet."

    def train(self, epochs=None):
        if epochs:
            self.hparams.epochs = epochs
            
        if self.hparams.dataset != None:
            print("Running training loop.")
            self.reports, self.shap_explanations = self.hparams.training_regime.train(self.hparams)
        else:
            print("No dataset provided. Please provide a dataset to train on.")
            return None
        return self.reports, self.shap_explanations

    def get_reports(self):
        return self.reports
    
    def list_runs(self):
        return self.reports.keys()
