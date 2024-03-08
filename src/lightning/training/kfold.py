import argparse
from os import path

import lightning as pl
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from lightning import seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, Subset
from torchmetrics.regression import MeanAbsoluteError

from sklearn.model_selection import KFold


from copy import deepcopy
import pandas as pd
import numpy as np
from src.project_system import project

# adapted from https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/fabric/kfold_cv/train_fabric.py

# class KFold:

def train_dataloader(model, data_loader, optimizer, fabric, epoch, hparams, fold, logger):
    # TRAINING LOOP
    model.train()
    
    running_loss = 0.0
    i = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        # NOTE: no need to call `.to(device)` on the data, target
        i += 1
        optimizer.zero_grad()
        output = model(data)
               # Ensure output and target have the same dimensions
        if output.dim() > 1:  # If output has more than one dimension
            output = torch.squeeze(output)
        if target.dim() == 1 and output.dim() == 0:  # If target is [batch_size,] and output is scalar
            target = target.unsqueeze(-1)  # Add a dimension to target to match output

        loss = hparams.loss_function(output, target)
        fabric.backward(loss)  # instead of loss.backward()
        running_loss = running_loss + loss.item()
        optimizer.step()
        if (batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):
            print(
                "Fold {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    fold,
                    epoch,
                    batch_idx * len(data),
                    len(data_loader.dataset),
                    100.0 * batch_idx / len(data_loader),
                    loss.item(),
                )
            )
        
        if hparams.dry_run:
            break
        
    running_loss = running_loss / i
    weight_norm = torch.norm(model.fc1.weight.data, p='fro')

    logger.log_metrics({'train_loss': running_loss}, step=epoch)
    logger.log_metrics({'weight_norm': weight_norm.item()}, step=epoch)

def validate_dataloader(model, data_loader, fabric, hparams, fold, acc_metric, logger, epoch):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            # NOTE: no need to call `.to(device)` on the data, target
            output = model(data)
            # Ensure output and target have the same dimensions
            # Ensure output has at least 1 dimension
            if output.dim() == 0:  # Output is a scalar
                output = output.unsqueeze(-1)  # Add a dimension to make it [1]
            
            # Similarly ensure target has at least 1 dimension
            if target.dim() == 0:
                target = target.unsqueeze(-1)

            # If output is not scalar but has more dimensions than target, squeeze it
            if output.dim() > target.dim():
                output = torch.squeeze(output)

            # Ensure target is reshaped to match output if necessary
            if target.dim() > output.dim():
                target = torch.squeeze(target)
            if target.dim() < output.dim():
                target = target.unsqueeze(-1)  # Add dimension to target to match output

            

            loss += hparams.loss_function(output, target).item()

            # Accuracy with torchmetrics
            acc_metric.update(output, target)

            if hparams.dry_run:
                break

    # all_gather is used to aggregate the value across processes
    loss = fabric.all_gather(loss).sum() / len(data_loader.dataset)

    # compute acc
    acc = acc_metric.compute()
    logger.log_metrics({'val_loss': loss, 'val_accuracy': acc}, step=epoch)

    print(f"\nFor fold: {fold} Validation set: Average loss: {loss:.4f}, MAE: ({acc:.0f})\n")
    return acc


def shap_plot(shap_explanation, show=True):
    import shap
    plot_obj = shap.waterfall_plot(shap_explanation, max_display=10, show=show)
    # plot_obj = plt.gcf()
    return plot_obj

def evaluate_model(model, dataset, train_loader, test_loader, val_ids, hparams, fold):
    explainer = hparams.feature_eval(dataset)
    report = explainer.evaluate(model, train_loader, test_loader, val_ids, hparams, fold)
    return report


def train(hparams):
    # Create the Lightning Fabric object. The parameters like accelerator, strategy, devices etc. will be proided
    # by the command line. See all options: `lightning run model --help`
    fabric = Fabric()

    seed_everything(hparams.seed)  # instead of torch.manual_seed(...)

    # Loop over different folds (shuffle = False by default so reproducible)
    if hparams.loocv:
        folds = len(hparams.dataset)
    else:
        folds = hparams.folds
    
    kfold = KFold(n_splits=folds)

    dataset = TensorDataset(torch.tensor(hparams.dataset.X.values, dtype=torch.float32), torch.tensor(hparams.dataset.y.values, dtype=torch.float32))


    project_path = project.set_project(hparams.project_name)
    experiment_name = hparams.experiment_name
    run_name = hparams.run_name

    if folds > len(dataset):
        raise ValueError(f"Number of folds {folds} is greater than the number of samples in the dataset {len(dataset)}")
    
    
    
    
    
    if folds == len(dataset) or hparams.loocv ==True:
        print(f"Running Leave-One-Out Cross Validation with {folds} folds.")
        loggers = [TensorBoardLogger(f"projects/{hparams.project_name}/{hparams.experiment_name}/{hparams.run_name}/lightning_logs", name=f"{id}") for id in hparams.dataset.sample_ids]
    else:
        loggers = [TensorBoardLogger(f"projects/{hparams.project_name}/{hparams.experiment_name}/{hparams.run_name}/lightning_logs", name=f"fold_{i}") for i in range(folds)]

    # initialize n_splits models and optimizers
    models = [hparams.base_model(hparams.input_size, hparams.hidden_size) for _ in range(kfold.n_splits)]
    optimizers = [hparams.optimizer(model.parameters(), lr=hparams.lr) for model in models]
    
    # fabric setup for models and optimizers
    for i in range(kfold.n_splits):
        models[i], optimizers[i] = fabric.setup(models[i], optimizers[i])

    # Accuracy using torchmetrics
    acc_metric = hparams.acc_metric().to(fabric.device)
    loss_metric = hparams.loss_function
    # loop over epochs
    for epoch in range(1, hparams.epochs + 1):
        # loop over folds
        epoch_acc = 0
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"Working on fold {fold} train ids: {train_ids} val ids: {val_ids} - sample_id {hparams.dataset.sample_ids[val_ids]}")

            
            # initialize dataloaders based on folds
            batch_size = hparams.batch_size
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_ids))
            val_subset = Subset(dataset, val_ids)
            val_loader = DataLoader(dataset, batch_size=len(val_subset), sampler=SubsetRandomSampler(val_ids))

            # set up dataloaders to move data to the correct device
            train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

            # get model and optimizer for the current fold
            model, optimizer = models[fold], optimizers[fold]
            logger = loggers[fold]
            
            # train and validate
            train_dataloader(model, train_loader, optimizer, fabric, epoch, hparams, fold, logger)
            epoch_acc += validate_dataloader(model, val_loader, fabric, hparams, fold, acc_metric, logger, epoch)
            acc_metric.reset()

        # log epoch metrics
        print(f"Epoch {epoch} - Average acc: {epoch_acc / kfold.n_splits}")

        if hparams.dry_run:
            break

    # When using distributed training, use `fabric.save`
    # to ensure the current process is allowed to save a checkpoint
    if hparams.save_model:
        fabric.save(model.state_dict(), "test.pt")
        
        
    torch.set_printoptions(precision=10)
    reports = {}
    shap_explanations = {}
    print("Evaluating models...")
    

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        
        sub_id = hparams.dataset.sample_ids.iloc[val_ids].tolist()
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_ids))
        val_subset = Subset(dataset, val_ids)
        val_loader = DataLoader(val_subset, batch_size=len(val_subset))
        
        counter = sub_id[0] if hparams.loocv else fold
        
        print(f"Running eval on fold: {counter} - sample: {hparams.dataset.sample_ids[val_ids]}")
        
        reports[str(counter)], shap_explanations[str(counter)] = evaluate_model(models[fold].to("cpu"), hparams.dataset, train_loader, val_loader, val_ids, hparams, fold)

        # report = reports[str(fold)]
        # print("Report for fold: ", fold)
        # print(report)
        report = pd.concat(reports.values(), ignore_index=True)
    return report, shap_explanations


















# if __name__ == "__main__":
#     # Arguments can be passed in through the CLI as normal and will be parsed here
#     # Example:
#     # lightning run model image_classifier.py accelerator=cuda --epochs=3
#     parser = argparse.ArgumentParser(description="Fabric K-Fold Cross Validation Example")
#     parser.add_argument(
#         "--batch-size", type=int, default=3, metavar="N", help="input batch size for training (default: 3)"
#     )
#     parser.add_argument("--epochs", type=int, default=80, metavar="N", help="number of epochs to train (default: 80)")
#     parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")
#     parser.add_argument("--gamma", type=float, default=0, metavar="M", help="Learning rate step gamma (default: 0.7)")
#     parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
#     parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")
#     parser.add_argument(
#         "--log-interval",
#         type=int,
#         default=2,
#         metavar="N",
#         help="how many batches to wait before logging training status",
#     )
#     parser.add_argument("--folds", type=int, default=2, help="number of folds for k-fold cross validation")
#     parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
 

    
#     data = pd.read_excel("data/processed/jhu_wm_tracts_FSL_thr15_FINAL.xlsx")
    
#     from src.dataset.pandas_dataset import BasePandasDataset as bpd
#     from src.lightning.models.mlp_regressor import SimpleMLP as base_model
    
#     target = 'FM_AVERAGE'
#     features_dict = {
#         'CST': ['CST', '_MEAN']
#     }
    
#     pandas_dataset = bpd(dataframe = data, target = target, sample_id_col='SUB_ID', feature_filter=features_dict)
    
#     dataset = TensorDataset(torch.tensor(pandas_dataset.X.values, dtype=torch.float32), torch.tensor(pandas_dataset.y.values, dtype=torch.float32))
    
#     parser.add_argument("--dataset", type=bpd, default=dataset, help="The dataset to use for training")
#     parser.add_argument("--base_model", type=pl.LightningModule, default=base_model, help="The base model to use for training")
#     parser.add_argument("--hidden_size", type=int, default=50, help="Default hidden layer size for the model")
#     parser.add_argument("--input_size", type=int, default=len(pandas_dataset.X.columns), help="Number of input features for the model")
    
#     hparams = parser.parse_args()
#     run(hparams)