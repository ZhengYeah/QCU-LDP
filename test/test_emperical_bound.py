import pytest
import numpy as np
from src.samples_from_mechanism import samples_of_mechanism


@pytest.mark.parametrize("x, sample_num, name, epsilon", [([0,0.5], 100, "laplace", 5)])
def test_samples_from_mechanism(x, sample_num, name, epsilon):
    samples = samples_of_mechanism(x, sample_num, name, epsilon)
    assert len(samples) == sample_num
    assert len(samples[0]) == len(x)
    print(samples)
