import torch
import torch.nn as nn

def generate_data(n=1000):
    X = torch.rand(n, 2)
    # circle boundary (x+1)^2 + (y+1)^2 = 4(log(2) + 1)
    y = (((X[:, 0] + 1) ** 2 + (X[:, 1] + 1) ** 2) < 4 * (torch.log(torch.tensor(2.0)) + 1)).long()
    return X, y

def ffnn(input_size=2, hidden_size=2, output_size=2):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )

def train(model, X, y, epochs=3000, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, loss {loss.item()}')


def accuracy(model, X, y):
    y_pred = model(X)
    y_pred = y_pred.argmax(dim=1)
    acc = (y_pred == y).float().mean()
    print(f'Accuracy: {acc.item()}')

if __name__ == '__main__':
    X, y = generate_data()
    model = ffnn()
    train(model, X, y)
    torch.save(model, 'ffnn.pth')
    accuracy(model, X, y)