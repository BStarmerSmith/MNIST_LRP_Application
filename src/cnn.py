import numpy as np
import torch
import torch.nn as nn
from src.variables import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import os

MINST_MEAN, MINST_STANDARD_DIV = 0.1307, 0.3081

class CNN:

    def __init__(self, epochs=5, batch_size=100, learning_rate=0.001):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MINST_MEAN,), (MINST_STANDARD_DIV, ))])
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = ConvNet()
        self.train_dataset = torchvision.datasets.MNIST(root=DATA_DIRECTORY, train=True, transform=transform, download=True)
        self.test_dataset = torchvision.datasets.MNIST(root=DATA_DIRECTORY, train=False, transform=transform)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def trainModel(self):
        print("Training Model")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        total_step = len(self.train_loader)
        loss_list, acc_list = [], []
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # Run the forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 100 == 0: # Print status every 100 images
                    print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.5f}, Accuracy: {:.5f}%'
                          .format(epoch + 1, self.epochs, i + 1, total_step, loss.item(),
                                  (correct / total) * 100))
    
    def testModel(self):
        print("Testing on 10000 images.")
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model was: {}%'.format((correct / total) * 100))

    # This function is used to save the model
    def saveModel(self):
        model_dir = os.path.join(MODEL_DIRECTORY, MODEL_FILENAME)
        torch.save(self.model, model_dir)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, data):
        out = self.layer1(data)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
