import pytorch_lightning as pl
import torch.nn.functional as F
import torch

class LinearRegressionModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, l2_strength=0.001, name="LinearRegressor"):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, 1)  # Our model has one input and one output
        self.l2_strength = l2_strength
        self.name = name
        
    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        
        # L2 Regularization
        l2_reg = torch.tensor(0.).to(self.device)
        
        if self.l2_strength != 0:
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