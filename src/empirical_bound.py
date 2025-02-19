import torch
import torch.nn as nn
from src.ldp_mechanisms import PiecewiseMechanism, DiscreteMechanism, NoiseAddingMechanism


class EmpiricalBound(nn.Module):
    def __init__(self, model, x):
        super(EmpiricalBound, self).__init__()
        self.model = model
        self.x = x
        self.correct_class = self.model(self.x.unsqueeze(0)).argmax(dim=1)

    def empirical_rho(self, samples):
        with torch.no_grad():
            predictions = self.model(samples)
        predictions = predictions.argmax(dim=1)
        correct_num = (predictions == self.correct_class).sum().item()
        return correct_num / samples.size(0)

    def samples_of_mechanism(self, sample_num, mechanism, epsilon):
        assert mechanism in ["pm", "sw", "krr", "exp", "laplace", "gaussian"]
        private_values = self.x.item()
        samples = []
        if mechanism == "pm":
            for _ in range(sample_num):
                one_sample = [PiecewiseMechanism(x, epsilon).linear_perturbation() for x in private_values]
                samples.append(one_sample)
        elif mechanism == "sw":
            samples = [PiecewiseMechanism(x, epsilon).sw_linear() for x in private_values]
        elif mechanism == "krr":
            samples = [DiscreteMechanism(x, epsilon, 256).krr() for x in private_values]
        elif mechanism == "exp":
            samples = [DiscreteMechanism(x, epsilon, 256).exp_mechanism_loc([0, 1]) for x in private_values]
        elif mechanism == "laplace":
            samples = [NoiseAddingMechanism(x, epsilon).laplace_mechanism() for x in private_values]
        elif mechanism == "gaussian":
            samples = [NoiseAddingMechanism(x, epsilon).gaussian_mechanism() for x in private_values]

        assert len(samples) == sample_num
        return torch.tensor(samples)

