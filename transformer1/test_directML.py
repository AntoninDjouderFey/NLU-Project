# test_directml.py
import torch, torch_directml

adapters = torch_directml.device_count()
print(adapters)

dml = torch_directml.device(1)
x = torch.randn(1024, 1024).to(dml)
y = torch.randn(1024, 1024).to(dml)
z = x @ y
print("Device:", dml)
print("Result shape:", z.shape)


