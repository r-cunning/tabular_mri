import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAutoencoder(pl.LightningModule):
    def __init__(self, input_size, lr=0.001, l2_strength=0.001, name="SimpleAutoencoder"):
        super(SimpleAutoencoder, self).__init__()
        # L2 regularization strength
        self.l2_strength = l2_strength

        
        self.test_loss = None
        self.name = name
        self.lr = lr
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 5),
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 6),
            nn.ReLU(),
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, input_size),
            nn.ReLU()
        )
        
        
    def forward(self, x):
        x_latent = self.encoder(x)
        y_hat = self.decoder(x_latent)
        return y_hat
    
    def encode_data(self, loader, train=False):
        encoded_features = []
        
        if train:
            self.train()
            for batch in loader:
                x, _ = batch
                encoded = self.encoder(x)
                encoded_features.append(encoded)

        else:
            self.eval()
            with torch.no_grad():  # No need to track gradients
                for batch in loader:
                    x, _ = batch
                    encoded = self.encoder(x)  # Use the encoder to generate latent representations
                    encoded_features.append(encoded)
        
        return torch.cat(encoded_features, dim=0)
    
    def encode(self, x):
        self.eval()
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        self.train()
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)  # Mean Squared Error Loss
        # L2 Regularization
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param)**2

        loss = loss + self.l2_strength * l2_reg
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        x, _ = batch
        x_hat = self.forward(x)
        val_loss = F.mse_loss(x_hat, x)
        self.log('val_loss', val_loss)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.l1_loss(x_hat, x)
        self.log('test_loss', loss)  # Logging the test loss
        self.test_loss = loss.item()
        return loss


    def on_test_epoch_end(self):
        # This method will be called at the end of the test epoch
        # You can add any summarizing or logging operations here if needed
        return self.test_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    


class RegressionMLP(pl.LightningModule):
    def __init__(self, input_size, output_size, lr=0.001):
        super(RegressionMLP, self).__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(input_size, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, output_size)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.l1_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class CombinedModel(pl.LightningModule):
    def __init__(self, autoencoder, regressor, lr=0.001, l2_strength=0.001):
        super(CombinedModel, self).__init__()
        self.autoencoder = autoencoder
        self.regressor = regressor
        self.lr = lr
        self.l2_strength = l2_strength  
    def forward(self, x):
        latent = self.autoencoder.encoder(x)
        y_hat = self.regressor(latent)
        return y_hat
    
    def predict(self, x):
        latent = self.autoencoder.encoder(x)
        return self.regressor(latent)

    def training_step(self, batch, batch_idx):
        x, y = batch
        latent = self.autoencoder.encoder(x)
        y_hat = self.regressor(latent)
        loss = F.l1_loss(y_hat, y)
        
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
        latent = self.autoencoder.encoder(x)
        y_hat = self.regressor(latent)
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        latent = self.autoencoder.encoder(x)
        y_hat = self.regressor(latent)
        test_loss = F.l1_loss(y_hat, y)
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer