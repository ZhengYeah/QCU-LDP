import torch
import torch.nn as nn
import math


class ProbabilisticRobustRadius(nn.Module):
    def __init__(self, model, x,  omega, tau, p=float('inf')):
        super(ProbabilisticRobustRadius, self).__init__()
        self.model = model
        self.x = x
        self.omega = omega
        self.tau = tau
        self.p = p
        self.correct_class = self.model(self.x.unsqueeze(0)).argmax(dim=1)
        # robust radius and area
        self.robust_radius = None
        self.clipped_robust_area = None
        self.robust_hyper_rectangle = None

    def hoeffding_bound_sample(self):
        """
        Compute the number of samples needed for the Hoeffding bound and samples from [0,1]^d
        """
        sample_num = math.log(2 / self.omega) / (2 * self.tau ** 2)
        sample_num = int(sample_num)
        samples = torch.rand(sample_num, self.x.dim())
        return samples, sample_num

    def robust_testing_radius(self, radius):
        print(f"Testing radius {radius}")
        samples_01, sample_num = self.hoeffding_bound_sample()
        samples = samples_01 * radius + self.x
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
            if self.robust_testing_radius(mid):
                lower = mid
            else:
                upper = mid
        self.robust_radius = upper
        self.clipped_robust_area = [self.x - self.robust_radius, self.x + self.robust_radius]
        self.clipped_robust_area = [torch.clamp(x, 0, 1) for x in self.clipped_robust_area]
        return upper


    def form_hyper_rectangle(self, step_size, rate):
        """
        Find a robust hyper rectangle (randomized algorithm)
        """
        assert self.robust_radius is not None
        # initialize the rectangle, shape: [x-\theta, x+\theta]
        rect = self.clipped_robust_area

        # sample the direction and add step_size
        left_direction = torch.bernoulli(torch.ones_like(self.x) * rate)
        right_direction = torch.bernoulli(torch.ones_like(self.x) * rate)
        # update the rectangle
        rect[0] = rect[0] - left_direction * step_size
        rect[1] = rect[1] + right_direction * step_size
        # clip the rectangle
        start_rect = [torch.clamp(x, 0, 1) for x in rect]
        return rect

    def robust_testing_rectangle(self, rect):
        samples_01, sample_num = self.hoeffding_bound_sample()
        # map the samples to the rectangle
        assert samples_01.dim() == rect[0].dim()
        rect_dim_sizes = rect[1] - rect[0]
        samples = samples_01 * rect_dim_sizes + rect[0]
        assert rect[0] <= samples <= rect[1]

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
                rect = self.form_hyper_rectangle(step, rate)
                if self.robust_testing_rectangle(rect):
                    return rect
        print(f"Cannot find a valid rectangle")
