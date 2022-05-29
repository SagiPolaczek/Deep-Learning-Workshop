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
import torch.optim as optim

from models import LitModel, POCModel

DATA_PATH = "tac/data/student-mat-pass-or-fail.csv"
AVAILABLE_GPUS = min(1, torch.cuda.device_count())

train_params = {
    'batch_size' : 64 if AVAILABLE_GPUS else 16,
    'epochs' : 10,

}

# Initialize model
model = POCModel()

# Initialize Dataset & DataLoader
transform = transforms.Compose(
    [transforms.ToTensor(),
    ])

train_data = SchoolDataset(file_path=DATA_PATH, train=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=train_params['batch_size'])

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# Train the model
for epoch in range(train_params['epochs']):

    running_loss = 0.0
    items = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data['image']
        labels = data['label']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        items += 1
    
    print(f'epoch {epoch + 1}, loss: {(running_loss / items) :.4f}')

print('Finished Training')


#
