import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F



class SimpleMLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, l2_strength=0.001, name="SimpleMLP"):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Only one output neuron for regression
        # L2 regularization strength
        self.l2_strength = l2_strength
        
        self.test_loss = None
        self.name = name
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        # loss = F.mse_loss(y_hat, y)  # Mean Squared Error Loss
        # L2 Regularization
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param)**2

        loss = loss + self.l2_strength * l2_reg
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.l1_loss(y_hat, y)
        self.log('val_loss', val_loss)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log('test_loss', loss)  # Logging the test loss
        self.test_loss = loss.item()
        return loss

    def on_test_epoch_end(self):
        # This method will be called at the end of the test epoch
        # You can add any summarizing or logging operations here if needed
        return self.test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    @property
    def model_name(self):
        return self.name