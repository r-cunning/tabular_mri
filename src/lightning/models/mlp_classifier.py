import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLPClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, l2_strength=0.001):
        super(SimpleMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Adjusted for multiple classes
        self.l2_strength = l2_strength
        self.num_classes = num_classes  # Store the number of classes
        
        self.save_hyperparameters()  # Optional: saves input_size, hidden_size, num_classes, and l2_strength

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)  # Cross-Entropy Loss for classification
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param)**2
        loss += self.l2_strength * l2_reg
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer