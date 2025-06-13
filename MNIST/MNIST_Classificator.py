import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import tqdm

train_data = datasets.MNIST(root = './data', train = True, download = 'True', transform = transforms.ToTensor())
test_data = datasets.MNIST(root = './data', train = False, download = 'True', transform = transforms.ToTensor())

train_data, val_data = random_split(train_data, [0.7, 0.3])

train_loader = DataLoader(train_data, batch_size=16, shuffle= True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(28*28, 256)
        self.linear2 = nn.Linear(256, 10)
        self.act = nn.ReLU()
        
    def forward(self, x):
        out = self.flat(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return out


    
if __name__ == "__main__":
    model = CNN()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)

    epochs = 7

    for epoch in range(epochs):
        train_loss_value = 0
        validation_loss_value = 0
        true_answer= 0
        model.train()
        for img, label in (pbar := tqdm.tqdm(train_loader)):
            pred = model(img)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_value +=loss.item()
            loss_item = loss.item()
            true_answer += (pred.argmax(dim = 0) == label.argmax(dim=0)).sum().item()
            
            pbar.set_description(f"Loss value {loss_item:.4f}")
        print(f" Epoch {epoch+1}: Mean loss value {train_loss_value/len(train_data):.4f}, Mean Accuracy: {true_answer / len(train_data):.4f} on train data")

        model.eval()
        with torch.no_grad():
            true_answer= 0
            for img, label in val_loader:
                pred = model(img)
                loss = loss_fn(pred, label)

                validation_loss_value+=loss.item()
                loss_item = loss.item()

                true_answer += (pred.argmax(dim = 0) == label.argmax(dim=0)).sum().item()
        print(f" Epoch {epoch+1}: Mean loss value {validation_loss_value/len(val_data):.4f}, Mean Accuracy: {true_answer / len(val_data):.4f} on validation data")
    
    torch.save(model.state_dict(), "mnist_model_final.pt")

    model.eval()
    test_acc = 0
    test_loss = 0
    
    with torch.no_grad():
        for img, label in test_loader:
            pred = model(img)
            loss = loss_fn(pred, label)
            
            test_loss += loss.item()
    
    print(f"Test results: Mean loss {test_loss/len(test_loader):.4f}, Mean Accuracy: {test_acc/len(test_loader):.4f}")


