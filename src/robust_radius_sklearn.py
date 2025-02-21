import math
import numpy as np
import pandas as pd


class RobustRadiusSKLearn:
    def __init__(self, model, x, perturb_col, omega, tau, p=float('inf')):
        """
        Find a robust vector for a given scikit-learn model and input
        :param model: scikit-learn model
        :param x: (dataframe) input to the model, contains all columns
        :param perturb_col: (list) the names of the cols to be perturbed
        :param omega: confidence level
        :param tau: tolerance level
        :param p: norm
        """
        assert isinstance(x, pd.DataFrame) and len(x) == 1
        assert isinstance(perturb_col, list)
        super(RobustRadiusSKLearn, self).__init__()
        self.model = model
        self.x = x
        self.perturb_col = perturb_col
        self.omega = omega
        self.tau = tau
        self.p = p
        self.correct_class = self.model.predict(self.x)[0]
        # robust radius and area
        self.robust_radius = None
        self.clipped_robust_area = None
        self.robust_hyper_rectangle = None

    def _hoeffding_bound_sample(self):
        """
        Compute the number of samples needed for the Hoeffding bound and form samples from [0,1]^d
        """
        sample_num = math.log(2 / self.omega) / (2 * self.tau ** 2)
        sample_num = int(sample_num)
        samples = np.random.rand(sample_num, len(self.perturb_col))
        # transform to [-1, 1]
        samples = samples * 2 - 1
        return samples, sample_num

    def _robust_testing_radius(self, radius):
        print(f"Testing radius {radius}")
        samples_01, sample_num = self._hoeffding_bound_sample()
        # map the samples to the radius
        assert samples_01.shape[1] == len(self.perturb_col)
        samples = samples_01 * radius + self.x[self.perturb_col].values
        # clip the samples to the range [0, 1]
        samples = np.clip(samples, 0, 1)

        # form the samples dataframe: copy the original x sample_num times
        samples_df = pd.DataFrame(data=np.tile(self.x.values, (sample_num, 1)), columns=self.x.columns)
        samples_df[self.perturb_col] = samples
        predictions = self.model.predict(samples_df)
        correct_num = (predictions == self.correct_class).sum()
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
        self.clipped_robust_area = [self.x[self.perturb_col].values - upper, self.x[self.perturb_col].values + upper]
        self.clipped_robust_area = [np.clip(x, 0, 1) for x in self.clipped_robust_area]
        return upper

    # finding a robust hyper rectangle
    def _form_hyper_rectangle(self, step_size, rate):
        """
        Find a robust hyper rectangle (randomized algorithm)
        """
        assert self.robust_radius is not None
        # initialize the rectangle, shape: [lower, upper] * perturb_col
        rect = self.clipped_robust_area

        # sample the direction and add step_size
        left_direction = np.random.binomial(1, rate, len(self.perturb_col))
        right_direction = np.random.binomial(1, rate, len(self.perturb_col))
        # update the rectangle
        assert len(left_direction) == len(right_direction) == len(self.perturb_col)

        rect[0] = rect[0] - left_direction * step_size
        rect[1] = rect[1] + right_direction * step_size
        # clip the rectangle
        clipped_rect = [np.clip(x, 0, 1) for x in rect]
        self.robust_hyper_rectangle = clipped_rect
        return clipped_rect

    def _robust_testing_rectangle(self, rect):
        samples_01, sample_num = self._hoeffding_bound_sample()
        # map the samples to the rectangle
        assert samples_01.shape[1] == len(self.perturb_col)
        rect_dim_sizes = rect[1] - rect[0]
        samples = samples_01 * rect_dim_sizes + rect[0]
        assert rect[0] <= samples <= rect[1]

        # form the samples dataframe: copy the original x sample_num times
        samples_df = pd.DataFrame(data=np.tile(self.x.values, (sample_num, 1)), columns=self.x.columns)
        samples_df[self.perturb_col] = samples
        predictions = self.model.predict(samples_df)
        correct_num = (predictions == self.correct_class).sum()
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
