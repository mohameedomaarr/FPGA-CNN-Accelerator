
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from model import CNN
from dataset import train_loader, test_loader


# Select CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Create the CNN
model = CNN().to(device)


# Loss function
criterion = nn.CrossEntropyLoss()


# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training configuration
EPOCHS = 5


# Training loop
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Training Accuracy: {accuracy:.2f}%"
    )


# Evaluate on test dataset
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)


test_accuracy = 100 * correct / total

print("\nFinal Test Accuracy:", f"{test_accuracy:.2f}%")


# Save trained model
weights_dir = Path(__file__).resolve().parent / "weights"
weights_dir.mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    weights_dir / "cnn_mnist.pth"
)

print("Model saved successfully!")
