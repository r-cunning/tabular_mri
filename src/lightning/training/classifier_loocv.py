import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

from src.lightning.models.mlp_classifier import SimpleMLPClassifier


def LOOCV(dataframe, X, y, num_classes, hidden_size=50, epochs=60):
    seed_everything(42, workers=True)
    loo = LeaveOneOut()

    batch_size = 3
    dataset = TensorDataset(X, y)
    reports = {}

    for train_idx, val_idx in loo.split(X):
        val_index = val_idx[0]
        sub_id = dataframe.iloc[val_index]['SUB_ID']
        reports[sub_id] = {}

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size)
        val_loader = DataLoader(val_subset, batch_size=1)

        model = SimpleMLPClassifier(X.shape[1], hidden_size, num_classes=num_classes)
        trainer = pl.Trainer(max_epochs=epochs)

        print(f"Training model on {len(train_idx)} samples with {hidden_size} hidden units...")
        trainer.fit(model, train_loader)

        # Use the model to predict the left-out sample
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            x_val = X[val_index].unsqueeze(0)  # Add batch dimension
            prediction = model(x_val)
            predicted_class = torch.argmax(prediction, dim=1)
            correct = (predicted_class == y[val_index]).item()
            
        reports[sub_id]['X'] = X[val_index].tolist()
        reports[sub_id]['y'] = y[val_index].item()
        reports[sub_id]['prediction'] = predicted_class.item()
        reports[sub_id]['correct'] = correct

    dataframe_report = pd.DataFrame.from_dict(reports, orient='index')
    dataframe_report['SUB_ID'] = dataframe_report.index

    
    
    
    y_true = dataframe_report['y']
    y_pred = dataframe_report['prediction']
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'micro', 'macro', or 'weighted' for multi-class
    auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    
    
    reports['accuracy'] = np.round(accuracy, 3)
    reports['f1'] = np.round(f1, 3)
    reports['auc'] = np.round(auc, 3)
    reports['precision'] = np.round(precision, 3)
    reports['recall'] = np.round(recall, 3)
    

    
    return reports, dataframe_report