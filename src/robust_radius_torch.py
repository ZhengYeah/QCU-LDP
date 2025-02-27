import torch
import torch.nn as nn
import math
from copy import deepcopy

class RobustRadiusTorch:
    def __init__(self, model, x, merged_ind, unmerged_ind, merge_step, omega, tau):
        """
        Find a robust vector for a given PyTorch model and input
        :param model: pytorch model
        :param x: (2D tensor) input to the model
        :param merged_ind: (list of 2D tuple) the indexes of the dims to be perturbed
        :param omega: confidence level
        :param tau: tolerance level
        """
        assert isinstance(x, torch.Tensor)
        assert isinstance(merged_ind, list)
        assert isinstance(model, nn.Module)
        super(RobustRadiusTorch, self).__init__()
        self.model = model
        self.x = x
        # perturbation indexes
        rows_merged = torch.tensor([merged_ind[i][0] for i in range(len(merged_ind))])
        cols_merged = torch.tensor([merged_ind[i][1] for i in range(len(merged_ind))])
        self.merged_ind = torch.stack((rows_merged, cols_merged), dim=0)
        rows_unmerged = torch.tensor([unmerged_ind[i][0] for i in range(len(unmerged_ind))])
        cols_unmerged = torch.tensor([unmerged_ind[i][1] for i in range(len(unmerged_ind))])
        self.unmerged_ind = torch.stack((rows_unmerged, cols_unmerged), dim=0)
        self.merge_step = merge_step
        self.total_dims = len(merged_ind) + len(unmerged_ind)

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
        samples = torch.rand(sample_num, self.total_dims)
        return samples, sample_num

    def _form_samples_for_dims(self, radius):
        """
        Form samples for the perturbed dimensions
        """
        noise_01, sample_num = self._hoeffding_bound_sample()
        noise_pos_neg = noise_01 * 2 - 1
        samples = self.x.clone().detach().unsqueeze(0).repeat(sample_num, 1, 1)
        assert samples.dim() == 3
        # for the unmerged dims
        unmerged_rows, unmerged_cols = self.unmerged_ind[0], self.unmerged_ind[1]
        num_unmerged_dims = len(unmerged_rows)
        samples[:, unmerged_rows, unmerged_cols] = samples[:, unmerged_rows, unmerged_cols] + noise_pos_neg[:, :num_unmerged_dims] * radius
        # for the merged dims, we need to add the perturbation to the corresponding dims
        merged_rows, merged_cols = self.merged_ind[0], self.merged_ind[1]
        samples[:, merged_rows, merged_cols] = samples[:, merged_rows, merged_cols] + noise_pos_neg[:, num_unmerged_dims:] * radius
        # copy to neighboring dims
        for i in range(1, self.merge_step):
            samples[:, merged_rows + i, merged_cols] = samples[:, merged_rows, merged_cols]
            samples[:, merged_rows, merged_cols + i] = samples[:, merged_rows, merged_cols]
            samples[:, merged_rows + i, merged_cols + i] = samples[:, merged_rows, merged_cols]
        # clip the samples to the range [0, 1]
        samples = torch.clamp(samples, 0, 1)
        # NOTE: this is verified correct, you can draw the samples to see the perturbation
        return samples, sample_num

    def _robust_testing_radius(self, radius):
        print(f"Testing radius {radius}")
        samples, sample_num = self._form_samples_for_dims(radius)
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
        # shape of the clipped robust area (only perturbed dimensions)
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
