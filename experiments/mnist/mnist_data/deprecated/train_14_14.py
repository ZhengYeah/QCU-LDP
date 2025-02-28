import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

# Load the MNIST
mnist_data = torchvision.datasets.MNIST('../', download=True, train=True)
# transform each image to 14x14
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2))
])

if torch.cuda.is_available():
    print(f"CUDA available, using GPU")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define the data loader
mnist_data.transform = transform
data_loader = DataLoader(mnist_data, batch_size=100, shuffle=True)
# Define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = CNN().to(device)
# Define the loss function
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Train the model
for epoch in range(10):
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        y_pred = model(images)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, loss {loss.item()}')
# Save the model
torch.save(model, '../../cnn_mnist_7_7.pth')

# Test the model
mnist_data_test = torchvision.datasets.MNIST('../', download=True, train=False)
mnist_data_test.transform = transform
data_loader_test = DataLoader(mnist_data_test, batch_size=100, shuffle=True)
correct = 0
total = 0
with torch.no_grad():
    for images, labels in data_loader_test:
        images, labels = images.to(device), labels.to(device)
        y_pred = model(images)
        _, predicted = torch.max(y_pred, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {correct / total}')
