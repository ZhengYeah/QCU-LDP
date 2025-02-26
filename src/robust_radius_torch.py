from copy import deepcopy

import torch
import torch.nn as nn
import math
from copy import deepcopy

class ProbabilisticRobustRadius(nn.Module):
    def __init__(self, model, x, perturb_ind, omega, tau):
        """
        Find a robust vector for a given PyTorch model and input
        :param model: pytorch model
        :param x: (2D tensor) input to the model
        :param perturb_ind: (list of 2D tuple) the indexes of the dims to be perturbed
        :param omega: confidence level
        :param tau: tolerance level
        """
        assert isinstance(x, torch.Tensor) and len(x.shape) == 1
        assert isinstance(perturb_ind, list)
        assert isinstance(model, nn.Module)
        super(ProbabilisticRobustRadius, self).__init__()
        self.model = model
        self.x = x
        # perturbation indexes
        rows = [self.perturb_ind[i][0] for i in range(len(self.perturb_ind))]
        cols = [self.perturb_ind[i][1] for i in range(len(self.perturb_ind))]
        self.perturb_ind = zip(rows, cols)
        self.omega = omega
        self.tau = tau
        self.correct_class = self.model(self.x.unsqueeze(0)).argmax(dim=1)
        # robust radius and area
        self.robust_radius = None
        self.clipped_robust_area = None
        self.robust_hyper_rectangle = None


    def _hoeffding_bound_sample(self):
        """
        Compute the number of samples needed for the Hoeffding bound and samples from [0,1]^d
        """
        sample_num = math.log(2 / self.omega) / (2 * self.tau ** 2)
        sample_num = int(sample_num)
        samples = torch.rand(sample_num, len(self.perturb_ind))
        return samples, sample_num

    def _robust_testing_radius(self, radius):
        print(f"Testing radius {radius}")
        samples_01, sample_num = self._hoeffding_bound_sample()
        samples_pos_neg = samples_01 * 2 - 1
        # map the samples to the radius
        assert samples_pos_neg.shape[1] == len(self.perturb_col)
        # samples = samples_pos_neg * radius
        # add the perturbation to the corresponding dims
        samples = self.x.clone().detach()
        samples = samples.repeat(sample_num, 1)
        assert samples.shape[0] == sample_num
        for row, col in enumerate(self.perturb_ind):
            samples[:, row, col] = samples[:, row, col] + samples_pos_neg[:,:] * radius
        # clip the samples to the range [0, 1]
        samples = torch.clamp(samples, 0, 1).unsqueeze(1)
        samples = samples.to(self.x.device)
        with torch.no_grad():
            predictions = self.model(samples)
        predictions = predictions.argmax(dim=1)
        correct_num = (predictions == self.correct_class).sum().item()
        return 1 - correct_num / sample_num <= self.tau

    def binary_search(self, lower=0, upper=0.5, tol=1e-2):
        while upper - lower > tol:
            mid = (lower + upper) / 2
            if self._robust_testing_radius(mid):
                lower = mid
            else:
                upper = mid
        self.robust_radius = upper
        # shape of the clipped robust area (only perturbed dimensions), shape: [[lower_1, lower_2, ...], [upper_1, upper_2, ...]]
        rows, cols = zip(*self.perturb_ind)
        tmp_area = self.x[rows, cols] - upper, self.x[rows, cols] + upper
        # clip the area to [0, 1]
        self.clipped_robust_area = [torch.clamp(x, 0, 1) for x in tmp_area]
        # initialize the robust hyper rectangle
        self.robust_hyper_rectangle = deepcopy(self.clipped_robust_area)
        return upper


    def _form_hyper_rectangle(self, initial_rect, step_size, rate):
        """
        Find a robust hyper rectangle (randomized algorithm)
        """
        assert self.robust_radius is not None
        # sample the direction and add step_size
        left_direction = torch.bernoulli(torch.ones_like(self.x) * rate)
        right_direction = torch.bernoulli(torch.ones_like(self.x) * rate)
        # update the rectangle
        rect = deepcopy(initial_rect)
        rect[0] = rect[0] - left_direction * step_size
        rect[1] = rect[1] + right_direction * step_size
        # clip the rectangle
        clipped_rect = [torch.clamp(x, 0, 1) for x in rect]
        return clipped_rect

    def _robust_testing_rectangle(self, rect):
        samples_01, sample_num = self._hoeffding_bound_sample()
        # map the samples to the rectangle
        assert samples_01.dim() == rect[0].dim()
        rect_dim_sizes = rect[1] - rect[0]
        samples = samples_01 * rect_dim_sizes + rect[0]
        assert rect[0].all() <= samples.all() <= rect[1].all()

        # test the samples
        samples = samples.to(self.x.device)
        with torch.no_grad():
            predictions = self.model(samples)
        predictions = predictions.argmax(dim=1)
        correct_num = (predictions == self.correct_class).sum().item()
        # True if the robustness rectangle is valid
        return 1 - correct_num / sample_num <= self.tau

    def adjust_step_rate(self, list_step_and_rate):
        """
        Adjust the step size and rate for the rectangle
        """
        assert len(list_step_and_rate[0]) == 2
        for step, rate in list_step_and_rate:
            for _ in range(100):
                rect = self._form_hyper_rectangle(step, rate)
                if self._robust_testing_rectangle(rect):
                    return rect
        print(f"Cannot find a valid rectangle")
