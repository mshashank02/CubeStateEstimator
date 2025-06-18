import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTConfig
import wandb

# --- Configuration ---
DATA_DIR = "/home/shashank/CubeStateEstimator/pose_dataset_HandManipulateBlock_ContinuousTouchSensors-v1_620000/"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Init wandb ---
wandb.init(project="cube-pose-estimation", config={
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "img_size": IMG_SIZE
})

# --- Dataset ---
class PoseDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = os.path.join(DATA_DIR, os.path.basename(item['image']))
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        position = torch.tensor(item['position'], dtype=torch.float32)
        quaternion = torch.tensor(item['quaternion'], dtype=torch.float32)
        return image, position, quaternion

# --- Load metadata ---
with open(os.path.join(DATA_DIR, "poses.json")) as f:
    metadata = json.load(f)

# --- Train/test split ---
train_meta, test_meta = train_test_split(metadata, test_size=0.2, random_state=42)

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# --- Dataloaders ---
train_loader = DataLoader(PoseDataset(train_meta, transform), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(PoseDataset(test_meta, transform), batch_size=BATCH_SIZE)

# --- Model ---
class ViTPoseEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.head = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 7)  # 3 pos + 4 quat
        )

    def forward(self, x):
        x = self.vit(pixel_values=x).last_hidden_state[:, 0]  # CLS token
        out = self.head(x)
        position = out[:, :3]
        quaternion = out[:, 3:]
        quaternion = quaternion / quaternion.norm(dim=1, keepdim=True)  # Normalize quaternion
        return position, quaternion

model = ViTPoseEstimator().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# --- Training ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, pos_gt, quat_gt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, pos_gt, quat_gt = imgs.to(DEVICE), pos_gt.to(DEVICE), quat_gt.to(DEVICE)

        pos_pred, quat_pred = model(imgs)
        loss_pos = loss_fn(pos_pred, pos_gt)
        loss_quat = loss_fn(quat_pred, quat_gt)
        loss = loss_pos + loss_quat

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    wandb.log({"train_loss": avg_train_loss})

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for imgs, pos_gt, quat_gt in test_loader:
            imgs, pos_gt, quat_gt = imgs.to(DEVICE), pos_gt.to(DEVICE), quat_gt.to(DEVICE)
            pos_pred, quat_pred = model(imgs)
            loss = loss_fn(pos_pred, pos_gt) + loss_fn(quat_pred, quat_gt)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        wandb.log({"val_loss": avg_val_loss})

    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

# --- Save model ---
torch.save(model.state_dict(), "vit_pose_model.pth")
wandb.finish()
