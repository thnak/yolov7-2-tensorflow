import torch
torch.device('vulkan')
a = torch.__version__
b = torch.is_vulkan_available()

print(f'version: {a}, vulkan: {b}')