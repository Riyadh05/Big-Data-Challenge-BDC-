import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import warnings
import os
warnings.filterwarnings("ignore")

from dataset import WasteDataset
from model import get_model

# ================== CONFIG ==================
DATA_DIR = '../data/train' # asumsikan dijalankan dari folder src/ atau root, kita pakai path yang aman
# Kalau error path, gunakan path absolut atau sesuaikan
if not os.path.exists(DATA_DIR):
    DATA_DIR = 'data/train'

MODEL_PATH = 'best_model.pth' # Sesuaikan dengan nama file model Anda
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = '../best_model.pth'

BATCH_SIZE = 16
MODEL_NAME = 'efficientnet_b0'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan device: {DEVICE}")

# ================== TRANSFORM & DATASET ==================
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = WasteDataset(DATA_DIR, transform=None)
_, val_idx = train_test_split(range(len(full_dataset)), test_size=0.15, stratify=full_dataset.labels, random_state=42)

val_dataset = WasteDataset(DATA_DIR, transform=val_transform)
val_dataset.images = [full_dataset.images[i] for i in val_idx]
val_dataset.labels = [full_dataset.labels[i] for i in val_idx]

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ================== LOAD MODEL ==================
model = get_model(MODEL_NAME, 3).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Berhasil memuat model dari: {MODEL_PATH}")
except Exception as e:
    print(f"Error memuat model: {e}")
    exit(1)

model.eval()

# ================== EVALUASI ==================
all_preds = []
all_labels = []

print("Mengevaluasi model pada data validasi...")
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

macro_f1 = f1_score(all_labels, all_preds, average='macro')
acc = (sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)) * 100

print("-" * 30)
print(f"Validation Accuracy : {acc:.2f}%")
print(f"Macro F1 Score      : {macro_f1:.5f}")
print("-" * 30)
print("\nDetail Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Organic', 'Recyclable', 'Non-Recyclable'] if len(set(all_labels))==3 else None))
