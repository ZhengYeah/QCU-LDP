import torch
from src.robust_radius_torch import RobustRadiusTorch

# load the model
model = torch.load('ffnn.pth')
# hypothesis test parameters
omega = 0.05
tau = 0.05
ht_test = RobustRadiusTorch(model, torch.tensor([0.5, 0.5]), omega, tau)
# binary search for the robust radius
robust_radius = ht_test.binary_search(0, 1)
print(f'==' * 20)
print(f'Robust Radius: {robust_radius}')
print(f'==' * 20)
