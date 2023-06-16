# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from PIL import Image
import torch.utils.data as data
from torch.utils.data import (
    Dataset,
    DataLoader,
) 
from torchvision.transforms import Resize
import time
import matplotlib.pyplot as plt
import copy
from tensorboardX import SummaryWriter
from ..logger.visualization import TensorboardWriter
from data_loader.data_loaders import CustomDataset
import logging

log_dir = "./logs"
logger = logging.getLogger(__name__)
enabled = True

writer = TensorboardWriter(log_dir, logger, enabled)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 6
learning_rate = 0.001
batch_size = 128
num_epochs = 25

# Load Data
dataset = CustomDataset(
    csv_file="./data/datasets/data.csv",
    root_dir="./data/datasets/datafolder",
    transform=transforms.Compose([
        Resize((256, 256)),
        transforms.ToTensor()
    ])
)

# Dataset is actually a lot larger ~25k images, just took out 10 pictures
# to upload to Github. It's enough to understand the structure and scale
# if you got more images.
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

train_set, test_set = data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = torchvision.models.mobilenet_v3_small(pretrained=True)

# final layer is not frozen
model.classifier[3] = nn.Linear(in_features=1024, out_features=num_classes)
# freeze all layers except final linear layer

model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Train Network
for epoch in range(num_epochs):
    since = time.time()
    losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward() 

        # gradient descent or adam step
        optimizer.step()

        # Log the average training loss
    avg_loss = sum(losses) / len(losses)
    # writer.add_scalar('Loss/train', avg_loss, epoch)
            # Evaluate the model on the validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            _, predictions = scores.max(1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    acc = correct / total
    # Log the validation accuracy
    writer.writer.add_scalar('Accuracy/val', acc, epoch)
    writer.writer.add_scalar('Loss/train', loss.item(), epoch)


     # Log the validation accuracy
    # writer.add_scalar('Accuracy/val', acc, epoch)


 # Save the weights of the model if its performance improved
 # Define the path to the 'model' folder
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
    if acc > best_acc:
        best_acc = acc
        best_model_wts = copy.deepcopy(model.state_dict())
        print("Best accuracy: ", best_acc)
         # Save the best accuracy and model to a file
         # Save the best accuracy and model to a file
        model_dir = os.path.join(MODEL_PATH, 'mobilenetv3')
        os.makedirs(model_dir, exist_ok=True)  # Create model1 folder if it doesn't exist
        model_file = os.path.join(model_dir, 'best.pt')
        torch.save({
            'accuracy': best_acc,
            'state_dict': best_model_wts
        }, model_file)

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Close the SummaryWriter
# writer.close()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
# load best model weights
model.load_state_dict(best_model_wts)

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

# Load the best model and evaluate its accuracy
# Define the path to the 'best.pt' file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
MODEL1_PATH = os.path.join(MODEL_PATH, 'mobilenetv3')
BEST_MODEL_FILE = os.path.join(MODEL1_PATH, 'best.pt')

# Load the model checkpoint
checkpoint = torch.load(BEST_MODEL_FILE)
model.load_state_dict(checkpoint['state_dict'])
check_accuracy(test_loader, model)