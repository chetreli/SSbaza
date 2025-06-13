import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Загрузка данных MNIST
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Извлечение данных в numpy массивы для совместимости с оригинальным кодом
x_train = train_dataset.data.numpy()
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

# Стандартизация входных данных
x_train = x_train / 255.0
x_test = x_test / 255.0

# Отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

# Определение модели нейронной сети
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

model = MNISTNet()

# Вывод структуры сети
print(model)
print(f"Общее количество параметров: {sum(p.numel() for p in model.parameters())}")

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Преобразование данных в тензоры PyTorch
x_train_tensor = torch.FloatTensor(x_train).unsqueeze(1)  # добавляем размерность канала
y_train_tensor = torch.LongTensor(y_train)
x_test_tensor = torch.FloatTensor(x_test).unsqueeze(1)
y_test_tensor = torch.LongTensor(y_test)

# Разделение на обучающую и валидационную выборки
val_size = int(0.2 * len(x_train_tensor))
train_size = len(x_train_tensor) - val_size

x_train_split = x_train_tensor[:train_size]
y_train_split = y_train_tensor[:train_size]
x_val_split = x_train_tensor[train_size:]
y_val_split = y_train_tensor[train_size:]

# Создание DataLoader'ов
train_dataset = TensorDataset(x_train_split, y_train_split)
val_dataset = TensorDataset(x_val_split, y_val_split)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Обучение модели
epochs = 5
model.train()

for epoch in range(epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
    
    # Валидация
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
    
    model.train()
    
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    
    print(f'Epoch [{epoch+1}/{epochs}], '
          f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

# Оценка на тестовой выборке
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += batch_y.size(0)
        test_correct += (predicted == batch_y).sum().item()

test_acc = 100 * test_correct / test_total
print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_acc:.2f}%')

# Предсказание для одного изображения
n = 1
x = x_test_tensor[n:n+1]  # берем одно изображение с сохранением размерности батча
model.eval()
with torch.no_grad():
    res = model(x)
    
print(res.numpy())
print(np.argmax(res.numpy()))

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Распознавание всей тестовой выборки
model.eval()
all_predictions = []

with torch.no_grad():
    for batch_x, _ in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.numpy())

pred = np.array(all_predictions)

print(pred.shape)
print(pred[:20])
print(y_test[:20])

# Выделение неверных вариантов
mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask]
y_false = y_test[~mask]  # исправлена ошибка из оригинального кода

print(x_false.shape)

# Вывод первых 25 неверных результатов
if len(x_false) >= 25:
    plt.figure(figsize=(10,5))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()
else:
    print(f"Найдено только {len(x_false)} неверных предсказаний")
    if len(x_false) > 0:
        plt.figure(figsize=(10,5))
        for i in range(len(x_false)):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x_false[i], cmap=plt.cm.binary)
        plt.show()