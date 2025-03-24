import math

import numpy as np
import random

pi = np.pi
exp = np.e


class PiecewiseMechanism:
    def __init__(self, private_val, epsilon):
        self.private_val = private_val
        self.epsilon = epsilon

    def linear_perturbation(self) -> "float":
        """
        the linear perturbation mechanism for [0,1]
        :return: the perturbed value
        """
        assert 0 <= self.private_val <= 1 + 1e-6
        C = (exp ** (self.epsilon / 2) - 1) / (2 * exp ** self.epsilon - 2)
        assert 0 < C
        p = exp ** (self.epsilon / 2)
        tmp = random.uniform(0, 1)
        if tmp < 2 * C * p:
            if C <= self.private_val <= 1 - C:
                sampled = random.uniform(self.private_val - C, self.private_val + C)
            elif self.private_val < C:
                sampled = random.uniform(0, 2 * C)
            else:
                sampled = random.uniform(1 - 2 * C, 1)
        else:
            if C <= self.private_val <= 1 - C:
                left_proportion = (self.private_val - C) / (1 - 2 * C)
                sampled = random.uniform(0, self.private_val - C) if random.uniform(0, 1) <= left_proportion else random.uniform(self.private_val + C, 1)
            elif self.private_val < C:
                sampled = random.uniform(2 * C, 1)
            else:
                sampled = random.uniform(0, 1 - 2 * C)
        return sampled

    def sw_linear(self) -> "float":
        """
        the redesigned square wave mechanism for linear perturbation
        """
        assert 0 <= self.private_val <= 1 + 1e-6
        p = (exp ** self.epsilon - 1) / self.epsilon
        C = (exp ** self.epsilon * (self.epsilon - 1) + 1) / (2 * (exp ** self.epsilon - 1) ** 2)
        tmp = random.uniform(0, 1)
        if tmp < p * 2 * C:
            if C <= self.private_val <= 1 - C:
                sampled = random.uniform(self.private_val - C, self.private_val + C)
            elif self.private_val < C:
                sampled = random.uniform(0, 2 * C)
            else:
                sampled = random.uniform(1 - 2 * C, 1)
        else:
            if C <= self.private_val <= 1 - C:
                left_proportion = (self.private_val - C) / (1 - 2 * C)
                sampled = random.uniform(0, self.private_val - C) if random.uniform(0, 1) <= left_proportion else random.uniform(self.private_val + C, 1)
            elif self.private_val < C:
                sampled = random.uniform(2 * C, 1)
            else:
                sampled = random.uniform(0, 1 - 2 * C)
        return sampled


class DiscreteMechanism:
    def __init__(self, private_val, epsilon, total_num: int):
        self.private_val = private_val
        self.epsilon = epsilon
        self.total_num = total_num

    def krr(self) -> "int":
        """
        the k-RR mechanism
        """
        assert 0 <= self.private_val <= 1
        private_index = math.floor(self.private_val * self.total_num)
        p = (1 / ((self.total_num - 1) + (exp ** self.epsilon))) * (self.total_num - 1)
        tmp = random.uniform(0, 1)
        if tmp < p:
            while True:
                sampled = random.randint(0, self.total_num)
                if sampled != private_index:
                    return float(sampled / self.total_num)
        else:
            return self.private_val

    def exp_abs(self) -> "float":
        """
        the exponential mechanism with absolute score function
        :return: perturbed value in [0,1]
        """
        # score function array
        score_array = np.zeros(self.total_num)
        for i in range(self.total_num):
            score_array[i] = - abs(self.private_val - i / self.total_num)
        # sensitivity = max (|x-y|-|x-y'|) = 1. It should be 1.
        sensitivity = 1
        # probability array
        p_list = np.zeros(self.total_num)
        for i in range(self.total_num):
            p_list[i] = exp ** (self.epsilon * score_array[i] / (2 * sensitivity))
        p_list = p_list / sum(p_list)
        # assert abs(sum(p_list) - 1) < 1e-2
        # sample
        tmp = random.uniform(0, 1)
        index_perturbed = None
        for i in range(self.total_num):
            if tmp <= sum(p_list[:i + 1]):
                index_perturbed = i
                break
        return index_perturbed / self.total_num

class NoiseAddingMechanism:
    def __init__(self, private_val, epsilon):
        self.private_val = private_val
        self.epsilon = epsilon

    def laplace_mechanism(self) -> "float":
        """
        the Laplace mechanism + truncation
        """
        assert 0 <= self.private_val <= 1 + 1e-6
        b = 1 / self.epsilon
        sampled = self.private_val + np.random.laplace(0, b)
        if sampled < 0:
            return 0
        elif sampled > 1:
            return 1
        else:
            return sampled

    def laplace_with_fail(self) -> ("float", "int"):
        """
        the Laplace mechanism with failure
        fail_num: the number of failures, used to compare with the theoretical failure rate
        """
        assert 0 <= self.private_val <= 1 + 1e-6
        fail_num = 0
        b = 1 / self.epsilon
        sampled = self.private_val + np.random.laplace(0, b)
        while sampled < 0 or sampled > 1:
            fail_num += 1
            sampled = self.private_val + np.random.laplace(0, b)
        return sampled, fail_num

    def gaussian_mechanism(self) -> "float":
        """
        the Gaussian mechanism
        """
        assert 0 <= self.private_val <= 1 + 1e-6
        sigma = 1 / self.epsilon
        sampled = self.private_val + np.random.normal(0, sigma)
        if sampled < 0:
            return 0
        elif sampled > 1:
            return 1
        else:
            return sampled

    def gaussian_with_fail(self, delta=0.052) -> ("float", "int"):
        """
        the Gaussian mechanism with failure
        fail_num: the number of failures, used to compare with the theoretical failure rate
        """
        assert 0 <= self.private_val <= 1 + 1e-6
        fail_num = 0
        # sigma = math.sqrt(2 * math.log(1.25 / delta)) / self.epsilon
        sigma = (math.sqrt(2) / 2) * (math.sqrt(math.log(2 / delta) + self.epsilon) + math.sqrt(math.log(2 / delta))) / self.epsilon
        sampled = self.private_val + np.random.normal(0, sigma)
        while sampled < 0 or sampled > 1:
            fail_num += 1
            sampled = self.private_val + np.random.normal(0, sigma)
        return sampled, fail_num

# if __name__ == "__main__":
#     mechanism = PiecewiseMechanism(0, 1)
#     print(mechanism.linear_perturbation())