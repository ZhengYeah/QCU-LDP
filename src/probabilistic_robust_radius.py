import torch
import torch.nn as nn


class ProbabilisticRobustRadius(nn.Module):
    def __init__(self, model, x,  omega, tau, p=float('inf')):
        super(ProbabilisticRobustRadius, self).__init__()
        self.model = model
        self.x = x
        self.omega = omega
        self.tau = tau
        self.p = p
        self.correct_class = self.model(self.x.unsqueeze(0)).argmax(dim=1)

    def robust_testing(self, radius):
        # hoeffding bound
        sample_num = torch.log(2 / self.omega) / (2 * self.tau ** 2)
        sample_num = int(sample_num)
        samples = torch.rand(sample_num, self.x.dim())
        samples = samples * radius + self.x
        # clip the samples to the range [0, 1]
        samples = torch.clamp(samples, 0, 1)

        samples = samples.to(self.x.device)
        with torch.no_grad():
            predictions = self.model(samples)
        predictions = predictions.argmax(dim=1)
        correct_num = (predictions == self.correct_class).sum().item()
        return 1 - correct_num / sample_num <= self.tau

    def binary_search(self, lower, upper, tol=1e-2):
        while upper - lower > tol:
            mid = (lower + upper) / 2
            if self.robust_testing(mid):
                lower = mid
            else:
                upper = mid
        return upper

