import numpy as np
from scipy.stats import laplace

pi = np.pi
exp = np.e

class CDFForRectangle:
    def __init__(self, epsilon, x, rectangle):
        """
        :param epsilon: privacy parameter
        :param rectangle: one-dimensional rectangle
        """
        self.epsilon = epsilon
        self.discretization_level = 500
        assert 0 <= x <= 1
        self.x = x
        assert 0 <= rectangle[0] <= 1 and 0 <= rectangle[1] <= 1
        self.rectangle = rectangle
        self.bin_num = 256


    def pm(self):
        # PDF of the piecewise mechanism
        C = (exp ** (self.epsilon / 2) - 1) / (2 * exp ** self.epsilon - 2)
        p = exp ** (self.epsilon / 2)
        if 0 <= self.x < C:
            l, r = 0, 2 * C
        elif 1 - C <= self.x <= 1:
            l, r = 1 - 2 * C, 1
        else:
            l, r = self.x - C, self.x + C
        # p and endpoints list
        p_list = [p / exp ** self.epsilon, p, p / exp ** self.epsilon]
        length_list = [0, l, r, 1]

        # CDF list of the piecewise mechanism
        cdf_list = np.zeros(self.discretization_level)
        for i in range(self.discretization_level):
            theta = i / self.discretization_level
            if 0 <= theta < length_list[1]:
                cdf_list[i] = p_list[0] * theta
            elif length_list[1] <= theta < length_list[2]:
                cdf_list[i] = p_list[0] * length_list[1] + p_list[1] * (theta - length_list[1])
            elif length_list[2] <= theta < length_list[3]:
                cdf_list[i] = p_list[0] * length_list[1] + p_list[1] * (length_list[2] - length_list[1]) + p_list[2] * (theta - length_list[2])
            else:
                cdf_list[i] = 1
        return cdf_list

    def laplace(self):
        # CDF of the Laplace mechanism
        mu = self.x
        b = 1 / self.epsilon
        theta = np.linspace(0, 1, self.discretization_level)
        cdf_list = laplace.cdf(theta, mu, b)
        return cdf_list

    def sw(self):
        # PDF of the square wave mechanism
        p = (exp ** self.epsilon - 1) / self.epsilon
        C = (exp ** self.epsilon * (self.epsilon - 1) + 1) / (2 * (exp ** self.epsilon - 1) ** 2)
        if 0 <= self.x < C:
            l, r = 0, 2 * C
        elif 1 - C <= self.x <= 1:
            l, r = 1 - 2 * C, 1
        else:
            l, r = self.x - C, self.x + C
        # p and endpoints list
        p_list = [p / exp ** self.epsilon, p, p / exp ** self.epsilon]
        length_list = [0, l, r, 1]

        # CDF list of the square wave mechanism
        cdf_list = np.zeros(self.discretization_level)
        for i in range(self.discretization_level):
            theta = i / self.discretization_level
            if 0 <= theta < length_list[1]:
                cdf_list[i] = p_list[0] * theta
            elif length_list[1] <= theta < length_list[2]:
                cdf_list[i] = p_list[0] * length_list[1] + p_list[1] * (theta - length_list[1])
            elif length_list[2] <= theta < length_list[3]:
                cdf_list[i] = p_list[0] * length_list[1] + p_list[1] * (length_list[2] - length_list[1]) + p_list[2] * (theta - length_list[2])
            else:
                cdf_list[i] = 1
        return cdf_list

    def exp_abs(self):
        # score function array
        score_array = np.zeros(self.bin_num)
        for i in range(self.bin_num):
            score_array[i] = - abs(self.x - i / self.bin_num)
        sensitivity = 0.5
        # probability array
        p_list = np.zeros(self.bin_num)
        for i in range(self.bin_num):
            p_list[i] = exp ** (self.epsilon * score_array[i] / (2 * sensitivity))
        p_list = p_list / sum(p_list)
        # CDF list of the exponential mechanism
        cdf_list = np.zeros(self.bin_num)
        for i in range(self.bin_num):
            cdf_list[i] = sum(p_list[:i + 1])
        return cdf_list

    def exp_square(self):
        # score function array
        score_array = np.zeros(self.bin_num)
        for i in range(self.bin_num):
            score_array[i] = - abs(self.x - i / self.bin_num) ** 2
        sensitivity = 0.5
        # probability array
        p_list = np.zeros(self.bin_num)
        for i in range(self.bin_num):
            p_list[i] = exp ** (self.epsilon * score_array[i] / (2 * sensitivity))
        p_list = p_list / sum(p_list)
        # CDF list of the exponential mechanism
        cdf_list = np.zeros(self.bin_num)
        for i in range(self.bin_num):
            cdf_list[i] = sum(p_list[:i + 1])
        return cdf_list

    def krr(self):
        # pdf list
        p = exp ** self.epsilon / ((self.bin_num - 1) + (exp ** self.epsilon))
        pdf_list = np.zeros(self.bin_num)
        for i in range(self.bin_num):
            index = self.x * self.bin_num
            pdf_list[i] = p if i == index else p / exp ** self.epsilon
        # cdf list
        cdf_list = np.zeros(self.bin_num)
        for i in range(self.bin_num):
            cdf_list[i] = sum(pdf_list[:i + 1])
        return cdf_list
