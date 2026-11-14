# train.py

import os
import sys
from pathlib import Path

# 1. Add project root to sys.path FIRST before any local imports
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

# 2. Now import safely after path is registered
from src.data.dataset import PCAPImageDataset

def set_seed(seed=42):
    """Sets seeds for complete reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model():
    # Fix random seed for reproducibility (as promised in README)
    set_seed(42)

    # Hyperparameters & Setup
    data_dir = "data/raw"
    batch_size = 32
    epochs = 5
    lr = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader (Excluding hold-out class from training)
    print("Loading dataset from PCAPs...")
    dataset = PCAPImageDataset(root_dir=data_dir, exclude_classes=["DictionaryBruteForce"])
    if len(dataset) == 0:
        print("Error: No samples found! Check your data/raw/ folders and ensure .pcap files are present.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    num_classes = len(dataset.classes)
    print(f"Total samples: {len(dataset)} | Number of classes: {num_classes}")
    print(f"Class mapping: {dataset.class_to_idx}")

    # Model Initialization
    model = timm.create_model('resnet18', pretrained=False, in_chans=3, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    print("\nStarting training loop...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc * 100:.2f}%")

    # Save Trained Model Weights along with Metadata
    os.makedirs("results/metrics", exist_ok=True)
    model_path = "results/metrics/resnet18_pcap_model.pth"
    
    checkpoint = {
        "model": "resnet18",
        "in_chans": 3,
        "class_to_idx": dataset.class_to_idx,
        "classes": dataset.classes,
        "state_dict": model.state_dict(),
    }
    
    torch.save(checkpoint, model_path)
    print(f"\nTraining complete! Model and metadata saved successfully to {model_path}")

if __name__ == "__main__":
    train_model()