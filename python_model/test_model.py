
import torch
from model import CNN

# Create model
model = CNN()

# Simulate one MNIST image
input_image = torch.randn(1, 1, 28, 28)

# Run inference
output = model(input_image)

print("Input shape:", input_image.shape)
print("Output shape:", output.shape)
print("Predicted digit:", output.argmax(dim=1).item())
