"""
Estimate the volume of robust area of a classifier using Monte Carlo method.
Input space is [0, 1]^d, where d is the dimension of the input.
The volume of the robust area can be estimated by sampling points uniformly from the input space and checking if they are in the robust area.
"""
from typing import Tuple

import torch
import torch.nn as nn

exp = torch.exp(torch.tensor(1.0))

class MonteCarloEstimator:
    def __init__(self, model, x, epsilon, sample_num=10000):
        """
        Estimate the volume of the robust area of a classifier using Monte Carlo method
        :param model: pytorch model
        :param x: (tensor) input to the model
        :param sample_num: number of samples to be drawn from the input space
        """
        assert isinstance(x, torch.Tensor) and x.dim() == 2
        assert isinstance(model, nn.Module)
        self.model = model
        self.x = x
        self.epsilon = epsilon
        self.sample_num = sample_num

    def samples_from_input_space(self) -> torch.Tensor:
        """
        Draw samples from the input space [0, 1]^d
        :return: (tensor) samples drawn from the input space
        """
        samples = torch.rand(self.sample_num, self.x.shape[0], self.x.shape[1])
        return samples

    def correct_samples(self, samples=None) -> torch.Tensor:
        """
        Draw samples from the input space and check if they are in the robust area
        :return: (tensor) samples that are in the robust area
        """
        if samples is None:
            raise ValueError("Samples must be provided for checking if they are in the robust area.")
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(samples.unsqueeze(1).float())
        predictions = predictions.argmax(dim=1)
        correct_class = self.model(self.x.unsqueeze(0).unsqueeze(0).float()).argmax(dim=1).item()
        correct_samples = samples[predictions == correct_class]
        return correct_samples

    def volume_of_robust_area(self) -> float:
        """
        Estimate the volume of the robust area of the classifier using uniform sampling
        :return: estimated volume of the robust area
        """
        self.model.eval()
        correct_class = self.model(self.x.unsqueeze(0).unsqueeze(0).float()).argmax(dim=1).item()
        samples = self.samples_from_input_space()
        with torch.no_grad():
            predictions = self.model(samples.unsqueeze(1).float())
        predictions = predictions.argmax(dim=1)
        correct_samples = samples[predictions == correct_class]
        volume = correct_samples.shape[0] / self.sample_num
        return volume

    def target_pdf_at_y_pm(self, y, epsilon) -> torch.Tensor:
        """
        Evaluate the PDF of the piecewise mechanism at y
        :param x: (float) the original input
        :param y: (tensor) the perturbed input
        :param epsilon: (float) the privacy budget
        :return:
        """
        # PDF of the piecewise mechanism
        exp_val = torch.exp(torch.tensor(1.0, device=self.x.device, dtype=self.x.dtype))
        C = (exp_val ** (epsilon / 2) - 1) / (2 * exp_val ** epsilon - 2)
        p = exp_val ** (epsilon / 2)
        # The piecewise mechanism has three pieces: [0, l), [l, r), [r, 1]
        l = torch.where(self.x < C, torch.zeros_like(self.x), self.x - C)
        r = torch.where(self.x >= 1 - C, torch.ones_like(self.x), self.x + C)
        # evaluate the PDF at y, which is also importance sampling weight for the samples drawn from the input space
        pdf = torch.zeros_like(y)
        pdf[y < l] = p / exp_val ** epsilon
        pdf[(y >= l) & (y < r)] = p
        pdf[y >= r] = p / exp_val ** epsilon
        return pdf

    def target_pdf_at_y_sw(self, y, epsilon) -> torch.Tensor:
        """
        Evaluate the PDF of the Staircase mechanism at y
        :param x: (float) the original input
        :param y: (tensor) the perturbed input
        :param epsilon: (float) the privacy budget
        :return:
        """
        # PDF of the staircase mechanism
        exp_val = torch.exp(torch.tensor(1.0, device=self.x.device, dtype=self.x.dtype))
        p = (exp ** self.epsilon - 1) / self.epsilon
        C = (exp ** self.epsilon * (self.epsilon - 1) + 1) / (2 * (exp ** self.epsilon - 1) ** 2)
        # The staircase mechanism has two pieces: [0, l), [l, 1]
        l = torch.where(self.x < C, torch.zeros_like(self.x), self.x - C)
        r = torch.where(self.x >= 1 - C, torch.ones_like(self.x), self.x + C)
        # evaluate the PDF at y, which is also importance sampling weight for the samples drawn from the input space
        pdf = torch.zeros_like(y)
        pdf[y < l] = p / exp_val ** epsilon
        pdf[(y >= l) & (y < r)] = p
        pdf[y >= r] = p / exp_val ** epsilon
        return pdf

    def target_prob_at_y_krr(self, y, epsilon) -> torch.Tensor:
        """
        Evaluate the PDF of the krr mechanism at y
        :param x: (float) the original input
        :param y: (tensor) the perturbed input
        :param epsilon: (float) the privacy budget
        :return:
        """
        # PDF of the krr mechanism
        exp_val = torch.exp(torch.tensor(1.0, device=self.x.device, dtype=self.x.dtype))
        # Assume discretization level is 100 for the krr mechanism
        discretization_level = 100
        x_index = torch.floor(self.x * discretization_level).long()
        p = exp_val ** self.epsilon / (exp_val ** self.epsilon + discretization_level - 1)
        y_index = torch.floor(y * discretization_level).long()
        # Check if there are same y_index of a sample with another sample
        for i in range(y_index.shape[0]):
            for j in range(i + 1, y_index.shape[0]):
                # If all dimensions of the y_index of a sample are the same as another sample, then they are the same y_index
                if torch.all(y_index[i] == y_index[j]):
                    # Remove the probability of the same y_index of a sample with another sample, which is not possible in the krr mechanism
                    y_index[j] = -1
        # evaluate the PDF at y, which is also importance sampling weight for the samples drawn from the input space
        prob_y = torch.where(y_index == x_index, p, (1 - p) / (discretization_level - 1))
        return prob_y

    def quantification_error_bound(self) -> float:
        volume = self.volume_of_robust_area()
        samples = self.samples_from_input_space()
        correct_samples = self.correct_samples(samples)
        pdfs_at_y = self.target_pdf_at_y_pm(correct_samples, self.epsilon)
        # The theoretical accuracy is the expected value of the PDF at the correct samples
        theoretical_acc = torch.mean(pdfs_at_y * pdfs_at_y).item() * volume
        return theoretical_acc

    def quantification_error_bound_sw(self) -> float:
        volume = self.volume_of_robust_area()
        samples = self.samples_from_input_space()
        correct_samples = self.correct_samples(samples)
        pdfs_at_y = self.target_pdf_at_y_sw(correct_samples, self.epsilon)
        # The theoretical accuracy is the expected value of the PDF at the correct samples
        theoretical_acc = torch.mean(pdfs_at_y * pdfs_at_y).item() * volume
        return theoretical_acc


    # def quantification_error_bound_krr(self) -> float:
    #     samples = self.samples_from_input_space()
    #     correct_samples = self.correct_samples(samples)
    #     probs_at_y = self.target_prob_at_y_krr(correct_samples, self.epsilon)
    #     # Probability of each sample is the product of the probabilities from 7 * 7 dimensions
    #     prob_each_sample = torch.prod(probs_at_y, dim=1)
    #     # The theoretical accuracy is the expected value of the probability of each sample
    #     theoretical_acc = torch.sum(prob_each_sample).item()
    #     return theoretical_acc
