import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

from dataset import WasteDataset
from model import get_model

# Config
DATA_DIR = 'data/train'
BATCH_SIZE = 16          # kecil dulu karena CPU
NUM_EPOCHS = 5           # mulai kecil
LEARNING_RATE = 0.001
MODEL_NAME = 'efficientnet_b0'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Transform
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset
full_dataset = WasteDataset(DATA_DIR, transform=None)  # transform nanti
train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.15, stratify=full_dataset.labels, random_state=42)

train_dataset = WasteDataset(DATA_DIR, transform=train_transform)
train_dataset.images = [full_dataset.images[i] for i in train_idx]
train_dataset.labels = [full_dataset.labels[i] for i in train_idx]

val_dataset = WasteDataset(DATA_DIR, transform=val_transform)
val_dataset.images = [full_dataset.images[i] for i in val_idx]
val_dataset.labels = [full_dataset.labels[i] for i in val_idx]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model
model = get_model(MODEL_NAME, 3).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Training dimulai...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {running_loss/len(train_loader):.4f}')
    
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation Accuracy: {100 * correct / total:.2f}%\n')

torch.save(model.state_dict(), 'best_model.pth')
print("Training selesai! Model disimpan.")