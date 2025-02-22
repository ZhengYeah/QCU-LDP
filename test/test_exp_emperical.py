from src.cdf_ldp_mechanisms_at_x import CDFAtX

robust_rectangle = [[0, 0], [0.4934, 1]]
private_values = [0.22, 0.567444]
mechanism = "krr"
epsilon = 1


def test_krr_theo():
    prob_accumulated = 1
    for i, private_value in enumerate(private_values):
        cdf_at_x = CDFAtX(epsilon, private_value, bin_num=100)
        cdf_list = cdf_at_x._krr()
        print(cdf_list)