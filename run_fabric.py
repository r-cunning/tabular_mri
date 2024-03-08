

import argparse
from src.lightning.training import kfold
import pandas as pd
import torch
import lightning as pl
from torch.utils.data import TensorDataset
from lightning.fabric.loggers import TensorBoardLogger
import torch.nn.functional as F
from lightning.fabric import Fabric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.metric import Metric
import torch.optim as optim
from torch.optim import Optimizer
from src.lightning.evaluation.shapley import ShapDeepExplainer as shapley



# adapted from https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/fabric/kfold_cv/train_fabric.py

if __name__ == "__main__":
    # Arguments can be passed in through the CLI as normal and will be parsed here
    # Example:
    # lightning run model image_classifier.py accelerator=cuda --epochs=3
    parser = argparse.ArgumentParser(description="Fabric K-Fold Cross Validation Example")
    parser.add_argument(
        "--batch-size", type=int, default=3, metavar="N", help="input batch size for training (default: 3)"
    )
    parser.add_argument("--epochs", type=int, default=60, metavar="N", help="number of epochs to train (default: 80)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")
    parser.add_argument("--gamma", type=float, default=0, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    
    parser.add_argument("--loocv", type=bool, default=True, help="Leave one out cross validation.")

    parser.add_argument("--optimizer", type=Optimizer, default=optim.Adam, help="Optimizer.")
    parser.add_argument("--feature_eval", default=shapley, help="Feature importance analysis.")

    parser.add_argument("--loss_function", default=F.l1_loss, help="Loss metric for training and validation.")

    parser.add_argument("--acc_metric", type=Metric, default=MeanAbsoluteError, help="Accuracy metric for validation and testing.")    
    
    parser.add_argument("--folds", type=int, default=24, help="number of folds for k-fold cross validation")
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")


    
    data = pd.read_csv("data/processed/starter_data.csv")
    
    from src.dataset.pandas_dataset import BasePandasDataset as bpd
    from src.lightning.models.mlp_regressor import SimpleMLP as base_model
    
    target = 'FM_AVERAGE'
    features_dict = {
        'aff': ['AFFECTED'],
        'unaff': ['UNAFFECTED'],
        'asym': ['ASYMMETRY'],
        'tss': ['TSS'],
    }
    
    pandas_dataset = bpd(dataframe = data, target = target, sample_id_col='SUB_ID', feature_filter=features_dict)
    
    # dataset = TensorDataset(torch.tensor(pandas_dataset.X.values, dtype=torch.float32), torch.tensor(pandas_dataset.y.values, dtype=torch.float32))
    
    
    logger = TensorBoardLogger("lightning_logs", name="my_model")
    
    parser.add_argument("--dataset", type=bpd, default=pandas_dataset, help="The dataset to use for training")
    parser.add_argument("--base_model", type=pl.LightningModule, default=base_model, help="The base model to use for training")
    parser.add_argument("--hidden_size", type=int, default=50, help="Default hidden layer size for the model")
    parser.add_argument("--input_size", type=int, default=len(pandas_dataset.X.columns), help="Number of input features for the model")
    parser.add_argument("--logger", type=TensorBoardLogger, default=logger, help="Logger")

    hparams = parser.parse_args()
    # reports, shap_explanations = kfold.run(hparams)
    
    
    # print(reports["0"])
    
    


    hparams = parser.parse_args()
    reports, shap_explanations = kfold.train(hparams)
    
