import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models import LitModel, LeNet, POCModel
from datasets import SchoolDataset

import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
import torch.optim as optim

DATA_PATH = "tac/data/student-mat-pass-or-fail.csv"
AVAILABLE_GPUS = min(1, torch.cuda.device_count())

# Training Parameters
train_params = {
    "batch_size": 64 if AVAILABLE_GPUS else 5,
    "epochs": 30,
    "learning_rate": 1e-3,
    "optim_momentum": 0.9,
}

###############
#### TRAIN ####
###############

# Initialize model
# model = POCModel()
model = LeNet()

# Initialize Dataset & DataLoader
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_data = SchoolDataset(file_path=DATA_PATH, train=True, transform=transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size=train_params["batch_size"])

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=train_params["learning_rate"], momentum=train_params["optim_momentum"])

print("Starting Training")

# Train the model
for epoch in range(train_params["epochs"]):

    running_loss = 0.0
    items = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data["image"]
        labels = data["label"]

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

    print(f"epoch {epoch + 1}, training loss: {(running_loss / items) :.4f}")

print("Finished Training")

##############
#### EVAL ####
##############
print("Starting Evaluation")

test_data = SchoolDataset(file_path=DATA_PATH, train=False, transform=transform)
test_loader = DataLoader(test_data, shuffle=False, batch_size=train_params["batch_size"])

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        inputs = data["image"]
        labels = data["label"]
        # calculate outputs by running images through the network
        outputs = model(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy of the network on the {len(test_data)} test samples: {100 * correct // total} %")
