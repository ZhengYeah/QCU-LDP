from src.ldp_mechanisms import PiecewiseMechanism, DiscreteMechanism, NoiseAddingMechanism
import numpy as np


def samples_of_mechanism(private_values, sample_num, mechanism, epsilon):
    """
    generate samples from the mechanism
    :param private_values: (list) the private values
    :param sample_num: (int) the number of samples
    :param mechanism: (str) the mechanism name
    :param epsilon: (float) the privacy budget
    :return: (np.ndarray) the samples
    """
    assert mechanism in ["pm", "sw", "krr", "exp", "laplace", "gaussian"]
    samples = np.zeros((sample_num, len(private_values)))
    if mechanism == "pm":
        for i in range(sample_num):
            one_sample = [PiecewiseMechanism(x, epsilon).linear_perturbation() for x in private_values]
            samples[i] = one_sample
    elif mechanism == "sw":
        for i in range(sample_num):
            one_sample = [PiecewiseMechanism(x, epsilon).sw_linear() for x in private_values]
            samples[i] = one_sample
    elif mechanism == "krr":
        for i in range(sample_num):
            one_sample = [DiscreteMechanism(x, epsilon, 256).krr() for x in private_values]
            samples[i] = one_sample
    elif mechanism == "exp":
        for i in range(sample_num):
            # the number of bins is 100 for normal [0, 1] feature
            # for image data, the number of bins is 256, considering the pixel value range
            one_sample = [DiscreteMechanism(x, epsilon, 100).exp_abs() for x in private_values]
            samples[i] = one_sample
    elif mechanism == "laplace":
        fail_num = 0
        for i in range(sample_num):
            tmp = []
            for x in private_values:
                one_sample, fail = NoiseAddingMechanism(x, epsilon).laplace_with_fail()
                tmp.append(one_sample)
                fail_num += fail
            samples[i] = tmp
        return samples, fail_num
    elif mechanism == "gaussian":
        for i in range(sample_num):
            one_sample = [NoiseAddingMechanism(x, epsilon).gaussian_mechanism() for x in private_values]
            samples[i] = one_sample

    assert len(samples) == sample_num
    return samples
