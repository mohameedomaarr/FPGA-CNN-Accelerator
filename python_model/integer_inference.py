
import numpy as np
import torch

from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

from model import CNN
from dataset import test_loader


# ------------------------------------------------
# 1. Paths and configuration
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"

MODEL_PATH = WEIGHTS_DIR / "cnn_mnist.pth"
QUANT_PATH = WEIGHTS_DIR / "cnn_quantized.npz"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Trained model not found. Run train.py.")

if not QUANT_PATH.exists():
    raise FileNotFoundError("Quantized parameters not found. Run quantize.py.")


# ------------------------------------------------
# 2. Load floating-point model
# ------------------------------------------------

model = CNN()

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()


# ------------------------------------------------
# 3. Load quantized parameters
# ------------------------------------------------

with np.load(QUANT_PATH) as params:

    conv_w = params["conv_weights_int8"]
    conv_b = params["conv_bias_int32"]

    fc_w = params["fc_weights_int8"]
    fc_b = params["fc_bias_int32"]

    input_scale = float(params["input_scale"])
    conv_weight_scale = float(params["conv_weight_scale"])
    activation_scale = float(params["activation_scale"])
    fc_weight_scale = float(params["fc_weight_scale"])


# Convert parameters to INT64 for safe reference calculations
conv_w = conv_w.astype(np.int64)
conv_b = conv_b.astype(np.int64)

fc_w = fc_w.astype(np.int64)
fc_b = fc_b.astype(np.int64)

INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max


# ------------------------------------------------
# 4. Integer CNN inference
# ------------------------------------------------

def integer_inference(image):

    # Quantize input image
    input_q = np.rint(
        image / input_scale
    )

    input_q = np.clip(
        input_q, 0, 255
    ).astype(np.int64)

    # --------------------------------------------
    # Convolution
    # --------------------------------------------

    # Input: 1 x 28 x 28
    windows = sliding_window_view(
        input_q[0],
        window_shape=(3, 3)
    )

    # Each output filter performs 9 MAC operations
    conv_acc = np.einsum(
        "ijuv,ouv->oij",
        windows,
        conv_w[:, 0],
        dtype=np.int64,
        optimize=True
    )

    # Add convolution biases
    conv_acc += conv_b[:, None, None]

    # Check accumulator range
    if np.any(conv_acc < INT32_MIN) or np.any(conv_acc > INT32_MAX):
        raise OverflowError("Convolution INT32 overflow")

    # --------------------------------------------
    # ReLU and activation quantization
    # --------------------------------------------

    conv_relu = np.maximum(conv_acc, 0)

    conv_multiplier = (
        input_scale * conv_weight_scale / activation_scale
    )

    activation_unclipped = np.rint(
        conv_relu * conv_multiplier
    )

    # Count saturation events
    clipped = np.count_nonzero(
        activation_unclipped > 255
    )

    activation_q = np.clip(
        activation_unclipped, 0, 255
    ).astype(np.int64)

    # --------------------------------------------
    # Max Pooling: 2 x 2
    # --------------------------------------------

    pooled = activation_q.reshape(
        8, 13, 2, 13, 2
    ).max(axis=(2, 4))

    # --------------------------------------------
    # Flatten
    # Channel -> Row -> Column
    # --------------------------------------------

    features = pooled.reshape(1352)

    # --------------------------------------------
    # Fully Connected Layer
    # --------------------------------------------

    fc_acc = fc_w @ features + fc_b

    if np.any(fc_acc < INT32_MIN) or np.any(fc_acc > INT32_MAX):
        raise OverflowError("Fully connected INT32 overflow")

    # The largest score determines the prediction
    prediction = int(np.argmax(fc_acc))

    return prediction, fc_acc, clipped


# ------------------------------------------------
# 5. Compare float and integer CNN
# ------------------------------------------------

float_correct = 0
integer_correct = 0
matching_predictions = 0

total = 0
total_clipped = 0
total_activations = 0

with torch.no_grad():

    for images, labels in test_loader:

        # Floating-point model
        float_outputs = model(images)

        float_predictions = (
            float_outputs.argmax(dim=1).numpy()
        )

        labels_np = labels.numpy()

        # Integer model
        for i in range(images.size(0)):

            image = images[i].numpy()

            int_prediction, scores, clipped = (
                integer_inference(image)
            )

            float_prediction = int(float_predictions[i])
            true_label = int(labels_np[i])

            if float_prediction == true_label:
                float_correct += 1

            if int_prediction == true_label:
                integer_correct += 1

            if int_prediction == float_prediction:
                matching_predictions += 1

            total_clipped += clipped
            total_activations += 8 * 26 * 26

            total += 1

        if total % 1024 == 0:
            print(f"Processed {total} images...")


# ------------------------------------------------
# 6. Print verification results
# ------------------------------------------------

print("\n========== VERIFICATION RESULTS ==========")

print(
    f"Float CNN accuracy: "
    f"{100 * float_correct / total:.2f}%"
)

print(
    f"Integer CNN accuracy: "
    f"{100 * integer_correct / total:.2f}%"
)

print(
    f"Prediction agreement: "
    f"{100 * matching_predictions / total:.2f}%"
)

print(
    f"Activation clipping rate: "
    f"{100 * total_clipped / total_activations:.4f}%"
)

print("\nInteger inference verification completed!")
