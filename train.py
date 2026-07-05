"""
Deepfake Detector — Training Script
------------------------------------
Trains a binary image classifier: real (0) vs fake (1).

Approach: TRANSFER LEARNING.
Instead of building a CNN from scratch (which needs millions of images
to work well), we take a model already trained on ImageNet (ResNet18),
which already knows how to recognize edges, textures, and shapes, and
we just retrain its final layer to answer our specific question:
"real or fake?". This needs far less data and trains much faster.

Expected folder structure:
    data/train/real/*.jpg
    data/train/fake/*.jpg
    data/val/real/*.jpg
    data/val/fake/*.jpg
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# =========================================================
# 1. CONFIG
# =========================================================
DATA_DIR = "data"
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0003
IMAGE_SIZE = 224          # standard input size for ResNet
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "deepfake_model.pt"


# =========================================================
# 2. DATA — loading and augmentation
# =========================================================
# Neural nets expect input tensors normalized the same way the pretrained
# model originally saw during ImageNet training. Mismatching these
# numbers is a classic silent bug (model runs, but performs badly).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),      # augmentation: teaches the model
    transforms.RandomRotation(10),          # to not rely on exact orientation
    transforms.ToTensor(),                  # converts image (0-255) -> tensor (0-1)
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# No augmentation for validation — we want to measure real performance,
# not performance-under-random-flips.
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ImageFolder automatically reads the data/train/real and data/train/fake
# folders and assigns labels based on folder name (alphabetical -> 0, 1).
train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transform)

print("Class mapping:", train_dataset.class_to_idx)  # e.g. {'fake': 0, 'real': 1}

# DataLoader batches the dataset and shuffles it each epoch.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =========================================================
# 3. MODEL — pretrained ResNet18, adapted for binary output
# =========================================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all existing layers: their weights already encode useful general
# image features, and we don't want to destroy that during early training.
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer. The original outputs 1000 classes (ImageNet);
# we only need 1 output: the probability of being "fake".
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

model = model.to(DEVICE)


# =========================================================
# 4. LOSS FUNCTION AND OPTIMIZER
# =========================================================
# BCEWithLogitsLoss = Binary Cross-Entropy, for a single real/fake output.
# It combines a sigmoid + the loss calculation in one numerically stable step.
criterion = nn.BCEWithLogitsLoss()

# Only the new final layer's weights need updating (everything else is frozen).
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)


# =========================================================
# 5. TRAINING LOOP
# =========================================================
def run_epoch(loader, training):
    """Runs one full pass over the data. Shared by train and validation."""
    model.train() if training else model.eval()

    total_loss = 0
    correct = 0
    total = 0

    # torch.no_grad() during validation: skips gradient tracking, saving
    # memory and time, since we're not updating weights here.
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)  # shape: (batch, 1)

            if training:
                optimizer.zero_grad()  # clear gradients from the previous batch

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()   # compute gradients
                optimizer.step()  # update weights

            total_loss += loss.item() * images.size(0)

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


print(f"Training on: {DEVICE}")

best_val_accuracy = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )

    # Save the model only when it's the best we've seen so far.
    # This protects against overfitting in later epochs.
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  -> New best model saved ({val_acc:.4f} val accuracy)")

print(f"\nTraining complete. Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Model weights saved to: {MODEL_SAVE_PATH}")
