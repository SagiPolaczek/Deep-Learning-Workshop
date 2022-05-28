import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models import LitModel
from datasets import SchoolDataset

import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms

from models import LitModel

DATA_PATH = "tac/data/student-mat-pass-or-fail.csv"
AVAILABLE_GPUS = min(1, torch.cuda.device_count())

train_params = {
    'batch_size' : 64 if AVAILABLE_GPUS else 16,
    'epochs' : 10,

}

# Initialize model
model = LitModel()

# Initialize Dataset & DataLoader
train_data = SchoolDataset(file_path=DATA_PATH, train=True)
train_loader = DataLoader(train_data, batch_size=train_params['batch_size'])

# Initialize Trainer
trainer = Trainer(
    gpus=AVAILABLE_GPUS,
    max_epochs=train_params['epochs'],
    progress_bar_refresh_rate=20,
)

# Train the model
trainer.fit(model, train_loader)
