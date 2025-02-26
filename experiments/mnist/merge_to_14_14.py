import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

# Load the MNIST
mnist_data = torchvision.datasets.MNIST('mnist_data/', download=False, train=True)
# transform each image to 14x14
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2))
])

mnist_data.transform = transform
data_loader = DataLoader(mnist_data, batch_size=100, shuffle=True)
# Define the model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(196, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
# Define the loss function
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Train the model
for epoch in range(30):
    for images, labels in data_loader:
        optimizer.zero_grad()
        y_pred = model(images)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, loss {loss.item()}')
# Save the model
torch.save(model, 'ffnn_mnist_14_14.pth')

# Test the model
mnist_data_test = torchvision.datasets.MNIST('mnist_data/', download=True, train=False)
mnist_data_test.transform = transform
data_loader_test = DataLoader(mnist_data_test, batch_size=100, shuffle=True)
correct = 0
total = 0
with torch.no_grad():
    for images, labels in data_loader_test:
        y_pred = model(images)
        _, predicted = torch.max(y_pred, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {correct / total}')
