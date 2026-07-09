import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import os

from dataset import WasteDataset
from model import get_model

# Config
MODEL_PATH = 'best_adv_efficientnet_b0_v1.pth'
TEST_DIR = 'data/test'
SUBMISSION_PATH = 'submission_TimITSB1.csv'   
BATCH_SIZE = 32
MODEL_NAME = 'efficientnet_b0'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Transform untuk test (sama seperti validation)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Test Dataset
test_dataset = WasteDataset(root_dir=TEST_DIR, transform=test_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load Model
model = get_model(MODEL_NAME, num_classes=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Melakukan prediksi pada data test...")

predictions = []
filenames = []

with torch.no_grad():
    for images, fnames in test_dataset:  # karena is_test=True
        images = images.unsqueeze(0).to(DEVICE)  # tambah batch dimension
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted.item())
        filenames.append(fnames)

# Buat Submission
submission = pd.DataFrame({
    'id': filenames,
    'predicted': predictions
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file berhasil dibuat: {SUBMISSION_PATH}")
print(f"Jumlah prediksi: {len(submission)}")
print(submission.head())