
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import CNN
from dataset import train_dataset


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"

MODEL_PATH = WEIGHTS_DIR / "cnn_mnist.pth"

# Number of images used to calibrate activation range
CALIBRATION_SAMPLES = 10000
BATCH_SIZE = 128

# MNIST ToTensor() pixels are in [0, 1].
# We represent them as UINT8 values in [0, 255].
INPUT_SCALE = 1.0 / 255.0


def quantize_weights(weights):
    """Symmetric INT8 quantization."""

    max_abs = np.max(np.abs(weights))

    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / 127.0

    quantized = np.round(weights / scale)
    quantized = np.clip(quantized, -127, 127)

    return quantized.astype(np.int8), float(scale)


def quantize_bias(bias, scale):
    """Quantize bias into INT32."""

    quantized = np.rint(bias / scale)

    int32_info = np.iinfo(np.int32)

    if np.any(quantized < int32_info.min):
        raise OverflowError("Bias underflow")

    if np.any(quantized > int32_info.max):
        raise OverflowError("Bias overflow")

    return quantized.astype(np.int32)


# Check model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "cnn_mnist.pth not found. Run train.py first."
    )


# Load trained CNN
device = torch.device("cpu")

model = CNN().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()


# ------------------------------------------------
# 1. Calibrate convolution activation scale
# ------------------------------------------------

calibration_size = min(
    CALIBRATION_SAMPLES,
    len(train_dataset)
)

calibration_dataset = Subset(
    train_dataset,
    range(calibration_size)
)

calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

max_activation = 0.0

with torch.no_grad():

    for images, _ in calibration_loader:

        images = images.to(device)

        conv_output = model.conv1(images)

        relu_output = model.relu(conv_output)

        batch_max = relu_output.max().item()

        max_activation = max(
            max_activation,
            batch_max
        )


if max_activation <= 0:
    raise ValueError("Invalid calibration activation range")

ACTIVATION_SCALE = max_activation / 255.0


# ------------------------------------------------
# 2. Extract floating-point parameters
# ------------------------------------------------

conv_weights = (
    model.conv1.weight.detach().cpu().numpy()
)

conv_bias = (
    model.conv1.bias.detach().cpu().numpy()
)

fc_weights = (
    model.fc.weight.detach().cpu().numpy()
)

fc_bias = (
    model.fc.bias.detach().cpu().numpy()
)


# ------------------------------------------------
# 3. Quantize convolution layer
# ------------------------------------------------

conv_weights_int8, conv_weight_scale = (
    quantize_weights(conv_weights)
)

conv_bias_int32 = quantize_bias(
    conv_bias,
    INPUT_SCALE * conv_weight_scale
)


# ------------------------------------------------
# 4. Quantize fully connected layer
# ------------------------------------------------

fc_weights_int8, fc_weight_scale = (
    quantize_weights(fc_weights)
)

fc_bias_int32 = quantize_bias(
    fc_bias,
    ACTIVATION_SCALE * fc_weight_scale
)


# ------------------------------------------------
# 5. Check accumulator limits
# ------------------------------------------------

conv_bound = (
    9 * 255 * 127
    + int(np.max(np.abs(conv_bias_int32.astype(np.int64))))
)

fc_bound = (
    1352 * 255 * 127
    + int(np.max(np.abs(fc_bias_int32.astype(np.int64))))
)

int32_max = np.iinfo(np.int32).max

if conv_bound > int32_max or fc_bound > int32_max:
    raise OverflowError(
        "Worst-case accumulator bound exceeds INT32 range"
    )


# ------------------------------------------------
# 6. Save quantized parameters
# ------------------------------------------------

np.savez(
    WEIGHTS_DIR / "cnn_quantized.npz",

    conv_weights_int8=conv_weights_int8,
    conv_bias_int32=conv_bias_int32,

    fc_weights_int8=fc_weights_int8,
    fc_bias_int32=fc_bias_int32,

    input_scale=INPUT_SCALE,
    conv_weight_scale=conv_weight_scale,
    activation_scale=ACTIVATION_SCALE,
    fc_weight_scale=fc_weight_scale
)


# Export integer CSV files for inspection
np.savetxt(
    WEIGHTS_DIR / "conv_weights_int8.csv",
    conv_weights_int8.reshape(8, 9),
    delimiter=",",
    fmt="%d"
)

np.savetxt(
    WEIGHTS_DIR / "conv_bias_int32.csv",
    conv_bias_int32,
    fmt="%d"
)

np.savetxt(
    WEIGHTS_DIR / "fc_weights_int8.csv",
    fc_weights_int8,
    delimiter=",",
    fmt="%d"
)

np.savetxt(
    WEIGHTS_DIR / "fc_bias_int32.csv",
    fc_bias_int32,
    fmt="%d"
)


# Save quantization metadata
metadata = {
    "input_format": "UINT8",
    "weight_format": "INT8",
    "bias_format": "INT32",
    "accumulator_format": "INT32",
    "activation_format": "UINT8",

    "input_scale": INPUT_SCALE,
    "conv_weight_scale": conv_weight_scale,
    "activation_scale": ACTIVATION_SCALE,
    "fc_weight_scale": fc_weight_scale,

    "calibration_samples": calibration_size,
    "rounding": "numpy.rint (ties-to-even)",

    "conv_accumulator_bound": conv_bound,
    "fc_accumulator_bound": fc_bound
}

with open(
    WEIGHTS_DIR / "quantization_metadata.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(metadata, file, indent=4)


# ------------------------------------------------
# 7. Display results
# ------------------------------------------------

print("\nQUANTIZATION COMPLETED")

print("\nConvolution:")
print("Weights:", conv_weights_int8.shape)
print("Bias:", conv_bias_int32.shape)
print("Weight scale:", conv_weight_scale)

print("\nFully Connected:")
print("Weights:", fc_weights_int8.shape)
print("Bias:", fc_bias_int32.shape)
print("Weight scale:", fc_weight_scale)

print("\nInput scale:", INPUT_SCALE)
print("Activation scale:", ACTIVATION_SCALE)

print("\nConvolution accumulator bound:", conv_bound)
print("FC accumulator bound:", fc_bound)

print("\nAll quantized parameters exported!")
