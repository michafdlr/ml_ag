# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import lightning as L
#from lightning.pytorch.callbacks import ModelCheckpoint


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./mnist", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        self.mnist_test = datasets.MNIST(
            self.data_dir, transform=transforms.ToTensor(), train=False
        )
        self.mnist_predict = datasets.MNIST(
            self.data_dir, transform=transforms.ToTensor(), train=False
        )
        mnist_full = datasets.MNIST(
            self.data_dir, transform=transforms.ToTensor(), train=True
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=2, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_units: int=64, hidden_layers: int=2):
        super().__init__()
        layers = [nn.Linear(in_features=in_features, out_features=hidden_units), nn.ReLU()]
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        layers.append(nn.Linear(hidden_units, out_features))

        self.stack = nn.Sequential(
            *layers
        )
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.stack(x)

class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def _shared_step(self, batch):
        features, y_true = batch
        logits = self(features)
        loss = F.cross_entropy(logits, y_true)
        y_pred = torch.argmax(logits, dim=1)
        return loss, y_true, y_pred

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        loss, y_true, y_pred = self._shared_step(batch)
        self.train_acc(y_pred, y_true)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_true, y_pred = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(y_pred, y_true)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch):
        _, y_true, y_pred = self._shared_step(batch)
        self.test_acc(y_pred, y_true)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        return [optimizer], [scheduler]
