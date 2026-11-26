
import json
from pathlib import Path

import numpy as np
import torch

from model import CNN

# Project paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"

MODEL_PATH = WEIGHTS_DIR / "cnn_mnist.pth"

# Check that the trained model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Run train.py first."
    )

# Load trained model
device = torch.device("cpu")

model = CNN().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

# Extract parameters as NumPy arrays
conv_weights = (
    model.conv1.weight.detach().cpu().numpy().astype(np.float32)
)

conv_bias = (
    model.conv1.bias.detach().cpu().numpy().astype(np.float32)
)

fc_weights = (
    model.fc.weight.detach().cpu().numpy().astype(np.float32)
)

fc_bias = (
    model.fc.bias.detach().cpu().numpy().astype(np.float32)
)

# Print parameter dimensions
print("Convolution weights:", conv_weights.shape)
print("Convolution bias:", conv_bias.shape)

print("FC weights:", fc_weights.shape)
print("FC bias:", fc_bias.shape)

# Save original floating-point parameters
np.savez(
    WEIGHTS_DIR / "cnn_parameters_float32.npz",
    conv_weights=conv_weights,
    conv_bias=conv_bias,
    fc_weights=fc_weights,
    fc_bias=fc_bias
)

# Export CSV files
# Each convolution filter is stored as one row.
np.savetxt(
    WEIGHTS_DIR / "conv_weights.csv",
    conv_weights.reshape(8, 9),
    delimiter=",",
    fmt="%.9g"
)

np.savetxt(
    WEIGHTS_DIR / "conv_bias.csv",
    conv_bias,
    fmt="%.9g"
)

np.savetxt(
    WEIGHTS_DIR / "fc_weights.csv",
    fc_weights,
    delimiter=",",
    fmt="%.9g"
)

np.savetxt(
    WEIGHTS_DIR / "fc_bias.csv",
    fc_bias,
    fmt="%.9g"
)

# Save architecture metadata
metadata = {
    "input_shape": [1, 28, 28],
    "convolution_filters": 8,
    "kernel_size": [3, 3],
    "convolution_output_shape": [8, 26, 26],
    "pooling_output_shape": [8, 13, 13],
    "flatten_size": 1352,
    "output_classes": 10,
    "conv_weight_layout": "out_channel,in_channel,kernel_row,kernel_col",
    "fc_weight_layout": "output_neuron,input_feature",
    "fc_flatten_order": "channel,row,column",
    "parameter_format": "float32"
}

with open(
    WEIGHTS_DIR / "metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

print("\nAll parameters exported successfully!")
print("Output directory:", WEIGHTS_DIR)