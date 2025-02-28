import torch
import torch.nn as nn
import math
from copy import deepcopy


class RobustRadiusTorch:
    def __init__(self, model, x, omega, tau):
        """
        Find a robust vector for a given PyTorch model and input
        :param model: pytorch model
        :param x: (2D tensor) input to the model
        :param merged_ind: (list of 2D tuple) the indexes of the dims to be perturbed
        :param omega: confidence level
        :param tau: tolerance level
        """
        assert isinstance(x, torch.Tensor) and x.dim() == 2
        assert isinstance(model, nn.Module)
        super(RobustRadiusTorch, self).__init__()
        self.model = model
        self.x = x
        self.omega = omega
        self.tau = tau
        self.correct_class = self.model(self.x.unsqueeze(0).unsqueeze(0).float()).argmax(dim=1).item()
        # robust radius and area
        self.robust_radius = None
        self.clipped_robust_area = None
        self.robust_hyper_rectangle = None

    def _hoeffding_bound_sample(self, sample_dim):
        """
        Compute the number of samples needed for the Hoeffding bound and samples from [0,1]^d
        """
        sample_num = math.log(2 / self.omega) / (2 * self.tau ** 2)
        sample_num = int(sample_num)
        samples = torch.rand(sample_num, sample_dim)
        return samples, sample_num

    def _robust_testing_radius(self, radius):
        print(f"Testing radius {radius}")
        samples_01, sample_num = self._hoeffding_bound_sample(sample_dim=self.x.shape[0] * self.x.shape[1])
        samples_pos_neg = (samples_01 * 2 - 1).view(sample_num, self.x.shape[0], self.x.shape[1])
        samples = samples_pos_neg * radius + self.x
        # clip the samples to [0, 1]
        samples = torch.clamp(samples, 0, 1)
        samples = samples.to(self.x.device)
        # unsqueeze the samples for CNN
        samples = samples.unsqueeze(1).float()
        with torch.no_grad():
            predictions = self.model(samples)
        predictions = predictions.argmax(dim=1)
        correct_num = (predictions == self.correct_class).sum().item()
        return 1 - correct_num / sample_num <= self.tau

    def binary_search(self, lower=0, upper=1, tol=1e-2):
        while upper - lower > tol:
            mid = (lower + upper) / 2
            if self._robust_testing_radius(mid):
                lower = mid
            else:
                upper = mid
        self.robust_radius = upper
        robust_area_shape = self.x - upper, self.x + upper
        # clip the area to [0, 1]
        self.clipped_robust_area = [torch.clamp(x, 0, 1) for x in robust_area_shape]
        # initialize the robust hyper rectangle
        self.robust_hyper_rectangle = deepcopy(self.clipped_robust_area)
        return upper

    def _form_hyper_rectangle(self, initial_rect, step_size, rate):
        """
        Find a robust hyper rectangle (randomized algorithm)
        """
        assert self.robust_radius is not None
        # sample the direction
        left_direction = torch.bernoulli(torch.ones_like(self.x) * rate)
        right_direction = torch.bernoulli(torch.ones_like(self.x) * rate)
        rect = deepcopy(initial_rect)
        rect[0] = rect[0] - left_direction * step_size
        rect[1] = rect[1] + right_direction * step_size
        # clip the rectangle
        clipped_rect = [torch.clamp(x, 0, 1) for x in rect]
        return clipped_rect

    def _robust_testing_rectangle(self, rect):
        samples_01, sample_num = self._hoeffding_bound_sample(sample_dim=self.x.shape[0] * self.x.shape[1])
        samples_01 = samples_01.view(sample_num, self.x.shape[0], self.x.shape[1])
        # map the samples to the rectangle
        assert samples_01[0].dim() == rect[0].dim()
        rect_dim_sizes = rect[1] - rect[0]
        samples = samples_01 * rect_dim_sizes + rect[0]
        assert rect[0].all() <= samples.all() <= rect[1].all()
        # test the samples
        samples = samples.to(self.x.device)
        # unsqueeze the samples for CNN
        samples = samples.unsqueeze(1).float()
        with torch.no_grad():
            predictions = self.model(samples)
        predictions = predictions.argmax(dim=1)
        correct_num = (predictions == self.correct_class).sum().item()
        # True if the robustness rectangle is valid
        accuracy = correct_num / sample_num
        # print(f"Accuracy: {accuracy}")
        return 1 - accuracy <= self.tau

    def adjust_step_rate(self, list_step_and_rate):
        """
        Adjust the step size and rate for the rectangle
        For better accuracy, this function can be called multiple times
        """
        assert len(list_step_and_rate[0]) == 2
        best_rect = deepcopy(self.robust_hyper_rectangle)
        for step, rate in list_step_and_rate:
            for _ in range(100):
                initial_rect = best_rect
                rect = self._form_hyper_rectangle(initial_rect, step, rate)
                if self._robust_testing_rectangle(rect):
                    if torch.sum(rect[1] - rect[0]) > torch.sum(best_rect[1] - best_rect[0]):
                        best_rect = rect
        self.robust_hyper_rectangle = best_rect
        return best_rect
