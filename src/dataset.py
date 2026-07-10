import os
from torch.utils.data import Dataset
from PIL import Image

class WasteDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.images = []
        self.labels = []
        
        if not is_test:
            class_map = {
                '0_Recyclable': 0,
                '1_Electronic': 1,
                '2_Organic': 2
            }
            for class_name, label in class_map.items():
                class_dir = os.path.join(root_dir, class_name)
                if os.path.exists(class_dir):
                    files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    print(f"Found {len(files)} images in {class_name}")
                    for img_name in files:
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)
        else:
            test_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))], 
                              key=lambda x: int(x.split('.')[0]))
            for img_name in test_files:
                self.images.append(os.path.join(root_dir, img_name))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if not self.is_test:
            return image, self.labels[idx]
        return image, os.path.basename(img_path)