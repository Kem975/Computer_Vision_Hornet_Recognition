import torch
from torch.utils.data.dataloader import DataLoader 
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from customDataset import myDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device used : {torch.cuda.get_device_name(device)}")

torch.cuda.empty_cache()

#hyperparameters
learning_rate = 1e-4
batch_size = 15
num_epochs = 100

dataset_train = myDataset(csv_file = 'train_hornet.csv', root_dir = '../../data/dataset_train', transform = transforms.ToTensor())
dataset_test = myDataset(csv_file = 'test_hornet.csv', root_dir = '../../data/dataset_test', transform = transforms.ToTensor())
 
#train_set, test_set = torch.utils.data.random_split(dataset, [2000, 885])
train_loader = DataLoader(dataset = dataset_train, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(dataset = dataset_test, batch_size = batch_size, shuffle=True)

# load the pre-trained model
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)

#PATH = './state_dict_model_2.pt' # Notre sauvegarde d'un bon réseau déjà entraîné pour les frelons

#model.load_state_dict(torch.load(PATH))
model.to(device)

accur_train = []
accur_test = []

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
def train(model):
# Train Network
    for epoch in range(num_epochs):
        losses = []

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

        print(f"Cost at epoch {epoch+1} is {sum(losses) / len(losses)}")

        if epoch%5 == 0:
            print("Checking accuracy on Training Set")
            accur_train.append(check_accuracy(train_loader, model))
            print("Checking accuracy on Test Set")
            accur_test.append(check_accuracy(test_loader, model))
            




# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()


    for x, y in loader:
        x = x.to(device=device)
        y = y.to(device=device)

        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )

    return num_correct / num_samples * 100

train(model)


#print(model)
#torch.save(model.state_dict(), PATH)

print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)


