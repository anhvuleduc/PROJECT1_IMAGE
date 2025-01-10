from model import CNNLocationModel
from image_data import UrbanDataset
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  
import time
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--checkpoint_path", type=str, default="urban_model.pth", help="Path to save ckpt")
args = parser.parse_args()

print("Loading model ...")
model = CNNLocationModel()
print("Loading data ...")
urban_dataset = UrbanDataset(args.data_path)
print("Create DataLoader instance ...")

data_loader = DataLoader(urban_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Starting training ...")

for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0
    epoch_start_time = time.time()

    with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch") as batch_bar:
        for batch_idx, (img_tensor, ref_coords, centroid_coords) in enumerate(batch_bar):
            img_tensor = img_tensor.to(device).float()  
            ref_coords = ref_coords.to(device).float()
            centroid_coords = centroid_coords.to(device).float()

            optimizer.zero_grad()

            output = model(img_tensor).squeeze(1)
            pred_dlat, pred_dlong = output[:, 0], output[:, 1]
            pred_coords = centroid_coords + torch.stack((pred_dlat, pred_dlong), dim=1)
            loss = criterion(pred_coords, ref_coords)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_bar.set_postfix({"Batch Loss": loss.item()})

    avg_loss = total_loss / len(data_loader)
    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch [{epoch + 1}/{args.epochs}] completed in {epoch_duration:.2f}s, Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.checkpoint_path)
    print(f"Model saved to {args.checkpoint_path}")

print("Training completed!")

