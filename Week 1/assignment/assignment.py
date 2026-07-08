import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # FIXES: OMP: Error #15 Duplicate runtime crash

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ==========================================
# Step 1 — Load the data (& Validation Split)
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the full training dataset
full_train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset       = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Create a 10% validation split
val_size = int(0.1 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size

# Set a manual seed for reproducible splits
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print(f"Training samples   : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")
print(f"Test samples       : {len(test_dataset)}")
print("-" * 40)

# ==========================================
# Step 2 — Define your model
# ==========================================

class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),     
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),     
            nn.Linear(128, 10)   
        )

    def forward(self, x):
        return self.net(x)

model = FashionClassifier()

# ==========================================
# Step 3 — Train the model
# ==========================================

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
train_losses = []
val_losses = []

print("Starting training loop...")
for epoch in range(epochs):
    
    # -- Training Phase --
    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # -- Validation Phase --
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            loss = loss_fn(logits, labels)
            running_val_loss += loss.item()
            
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

print("-" * 40)

# ==========================================
# Step 4 — Plot the loss curve
# ==========================================

plt.figure(figsize=(9, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o', color='blue')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='s', color='red')
plt.title("FashionMNIST: Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Saves the plot directly to your directory for submission
plt.savefig("fashion_mnist_loss_curve.png", dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved to your current directory as 'fashion_mnist_loss_curve.png'.")
print("-" * 40)

# ==========================================
# Step 5 — Evaluate on the test set
# ==========================================

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100.0 * correct / total
print(f"Final Test Accuracy: {accuracy:.2f}%")