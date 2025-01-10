import torch
import torch.nn as nn

class ProbabilisticRobustRadius(nn.Module):
    def __init__(self, model, radius, p=2):
        super(ProbabilisticRobustRadius, self).__init__()
        self.model = model
        self.radius = radius
        self.p = p

    def forward(self, x):
        # get the prediction
        y_pred = self.model(x)
        # get the distance from the decision boundary
        dist = torch.norm(y_pred, p=self.p, dim=1)
        # get the probability
        prob = torch.exp(-dist / self.radius)
        return prob

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

