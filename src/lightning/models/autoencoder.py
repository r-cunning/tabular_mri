import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def build_sequential_layer(layer_sizes):
    layers = []
    for size in layer_sizes:
        layers.append(nn.Linear(size[0], size[1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class SimpleAutoencoder(pl.LightningModule):
    def __init__(self, encoder_layers, decoder_layers, lr=0.001, l2_strength=0.0001, name="SimpleAutoencoder"):
        super(SimpleAutoencoder, self).__init__()
        self.l2_strength = l2_strength
        self.name = name
        self.lr = lr
        
        self.encoder = build_sequential_layer(encoder_layers)
        self.decoder = build_sequential_layer(decoder_layers)

    def forward(self, x):
        x_latent = self.encoder(x)
        y_hat = self.decoder(x_latent)
        return y_hat
    
    def encode(self, x):
        self.eval()
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param)**2
        loss = loss + self.l2_strength * l2_reg
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        val_loss = F.mse_loss(x_hat, x)
        self.log('val_loss', val_loss)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.l1_loss(x_hat, x)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class RegressionMLP(pl.LightningModule):
    def __init__(self, layer_sizes, lr=0.001):
        super(RegressionMLP, self).__init__()
        self.lr = lr
        self.model = build_sequential_layer(layer_sizes)

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
    def __init__(self, autoencoder, regressor, lr=0.001, l2_strength=0.0001):
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