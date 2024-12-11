import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Step 1: Business Understanding
# Goal: Build a Dense NN and CNN to identify handwritten digits using the MNIST dataset.

# Step 2: Data Understanding
# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Visualize some examples
examples = iter(train_loader)
example_data, example_targets = next(examples)
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(example_data[i][0], cmap="gray")
    plt.title(f"Label: {example_targets[i].item()}")
    plt.axis("off")
plt.show()

# Step 3: Data Preparation
# Data is already normalized and in PyTorch format.

# Step 4: Modeling
# Define Dense NN Model
class DenseNN(nn.Module):
    def __init__(self):
        super(DenseNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*5*5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Step 5: Evaluation
# Training function
def train_model(model, train_loader, criterion, optimizer, device, history):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    history['train_loss'].append(running_loss / len(train_loader))
    history['train_accuracy'].append(accuracy)
    return running_loss / len(train_loader)

# Evaluation function
def evaluate_model(model, test_loader, criterion, device, history):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    history['test_loss'].append(test_loss / len(test_loader))
    history['test_accuracy'].append(accuracy)
    return test_loss / len(test_loader), accuracy

# Initialize models, optimizers, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dense_model = DenseNN().to(device)
cnn_model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
dense_optimizer = optim.Adam(dense_model.parameters(), lr=0.001)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Histories
dense_history = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}
cnn_history = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}

# Train and evaluate Dense NN
print("Training Dense Neural Network...")
for epoch in range(10):
    train_loss = train_model(dense_model, train_loader, criterion, dense_optimizer, device, dense_history)
    test_loss, accuracy = evaluate_model(dense_model, test_loader, criterion, device, dense_history)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Train and evaluate CNN
print("Training Convolutional Neural Network...")
for epoch in range(10):
    train_loss = train_model(cnn_model, train_loader, criterion, cnn_optimizer, device, cnn_history)
    test_loss, accuracy = evaluate_model(cnn_model, test_loader, criterion, device, cnn_history)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Visualize training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(dense_history['train_loss'], label='Dense NN Train Loss')
plt.plot(dense_history['test_loss'], label='Dense NN Test Loss')
plt.title('Dense NN Loss History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(dense_history['train_accuracy'], label='Dense NN Train Accuracy')
plt.plot(dense_history['test_accuracy'], label='Dense NN Test Accuracy')
plt.title('Dense NN Accuracy History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(cnn_history['train_loss'], label='CNN Train Loss')
plt.plot(cnn_history['test_loss'], label='CNN Test Loss')
plt.title('CNN Loss History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history['train_accuracy'], label='CNN Train Accuracy')
plt.plot(cnn_history['test_accuracy'], label='CNN Test Accuracy')
plt.title('CNN Accuracy History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()
