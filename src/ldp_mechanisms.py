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
        assert 0 <= self.private_val < self.total_num
        p = 1 / ((self.total_num - 1) + (exp ** self.epsilon))
        tmp = random.uniform(0, 1)
        if tmp < p:
            while True:
                sampled = random.randint(0, self.total_num)
                if sampled != self.private_val:
                    return sampled
        else:
            return self.private_val

    def exp_mechanism_loc(self, location_list: "list"):
        """
        the exponential mechanism for location perturbation
        """
        length = len(location_list)
        # corner case: only one location -> sensitivity is inf
        if length == 1:
            return location_list[0]
        # score function array
        score_array = np.zeros(length)
        for i in range(length):
            score_array[i] = -np.linalg.norm(np.array(location_list[i]) - np.array(self.private_val))
        sensitivity = max(score_array) - min(score_array)
        # probability array
        p = np.zeros(length)
        for i in range(length):
            p[i] = exp ** (self.epsilon * score_array[i] / (2 * sensitivity))
        p = p / sum(p)
        assert abs(sum(p) - 1) < 1e-3
        # sample
        tmp = random.uniform(0, 1)
        index_perturbed = None
        for i in range(length):
            if tmp <= sum(p[:i + 1]):
                index_perturbed = i
                break
        return location_list[index_perturbed]


class NoiseAddingMechanism:
    def __init__(self, private_val, epsilon):
        self.private_val = private_val
        self.epsilon = epsilon

    def laplace_mechanism(self) -> "float":
        """
        the Laplace mechanism
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

# if __name__ == "__main__":
#     mechanism = PiecewiseMechanism(0, 1)
#     print(mechanism.linear_perturbation())