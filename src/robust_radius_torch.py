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
        assert isinstance(x, torch.Tensor) and x.dim() == 2
        assert isinstance(merged_ind, list)
        assert isinstance(model, nn.Module)
        super(RobustRadiusTorch, self).__init__()
        self.model = model
        self.x = x
        # perturbation indexes
        # Shape of self.merged_ind: [2, rows & cols (each dim 0)]
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

    def _form_samples_for_dims(self, radius):
        """
        Form samples for the perturbed dimensions, sever for the method "_robust_testing_radius"
        """
        noise_01, sample_num = self._hoeffding_bound_sample(sample_dim=self.total_dims)
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
        # if the model has a flatten layer, there is no need to flatten the samples
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

    def _form_hyper_rectangle_for_dims(self, rate):
        """
        Form a robust hyper rectangle for the perturbed dimensions, sever for the method "_form_hyper_rectangle"
        """
        assert self.robust_radius is not None
        # sample the direction (NOTE: this is for all the dimensions, we will process the merged dims later)
        left_direction = torch.bernoulli(torch.ones_like(self.x) * rate)
        right_direction = torch.bernoulli(torch.ones_like(self.x) * rate)
        # process the merged dims
        for i in range(1, self.merge_step):
            # for the merged dims, we need to add the perturbation to the corresponding dims
            # NOTE: this is verified correct, you can draw the direction map to validate
            # Insight: from direction for all dims, and then unify the merged dims
            left_direction[self.merged_ind[0] + i, self.merged_ind[1]] = left_direction[self.merged_ind[0], self.merged_ind[1]]
            left_direction[self.merged_ind[0], self.merged_ind[1] + i] = left_direction[self.merged_ind[0], self.merged_ind[1]]
            left_direction[self.merged_ind[0] + i, self.merged_ind[1] + i] = left_direction[self.merged_ind[0], self.merged_ind[1]]
            right_direction[self.merged_ind[0] + i, self.merged_ind[1]] = right_direction[self.merged_ind[0], self.merged_ind[1]]
            right_direction[self.merged_ind[0], self.merged_ind[1] + i] = right_direction[self.merged_ind[0], self.merged_ind[1]]
            right_direction[self.merged_ind[0] + i, self.merged_ind[1] + i] = right_direction[self.merged_ind[0], self.merged_ind[1]]
        # the unmerged dims are fine
        return left_direction, right_direction

    def _form_hyper_rectangle(self, initial_rect, step_size, rate):
        """
        Find a robust hyper rectangle (randomized algorithm)
        """
        assert self.robust_radius is not None
        # sample the direction
        left_direction, right_direction = self._form_hyper_rectangle_for_dims(rate)
        rect = deepcopy(initial_rect)
        rect[0] = rect[0] - left_direction * step_size
        rect[1] = rect[1] + right_direction * step_size
        # clip the rectangle
        clipped_rect = [torch.clamp(x, 0, 1) for x in rect]
        return clipped_rect

    def _robust_testing_rectangle(self, rect):
        samples_01, sample_num = self._hoeffding_bound_sample(sample_dim=self.x.shape[0] * self.x.shape[1])
        samples_01 = samples_01.view(sample_num, self.x.shape[0], self.x.shape[1])
        # process the merged dims
        for i in range(1, self.merge_step):
            samples_01[:, self.merged_ind[0] + i, self.merged_ind[1]] = samples_01[:, self.merged_ind[0], self.merged_ind[1]]
            samples_01[:, self.merged_ind[0], self.merged_ind[1] + i] = samples_01[:, self.merged_ind[0], self.merged_ind[1]]
            samples_01[:, self.merged_ind[0] + i, self.merged_ind[1] + i] = samples_01[:, self.merged_ind[0], self.merged_ind[1]]
        # the unmerged dims are fine
        # NOTE: this is verified correct, you can draw the samples to validate
        # map the samples to the rectangle
        assert samples_01[0].dim() == rect[0].dim()
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
