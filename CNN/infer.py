import argparse
from model import CNNLocationModel
from image_data import UrbanDataset
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../assets/logFile_urban_data.csv", help="Path to the CSV file")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
parser.add_argument("--ckpt", type=str, default="urban_model.pth", help="Path to the model checkpoint")
parser.add_argument("--idx", type=int, required=True, help="Index of the data point in CSV file to infer")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLocationModel()
model.load_state_dict(torch.load(args.ckpt, map_location=device))
model.to(device)
model.eval()

urban_dataset = UrbanDataset(args.data_path)
data_loader = DataLoader(urban_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

with torch.no_grad():
    sample = urban_dataset[args.idx]
    img_tensor, ref_coords, centroid_coords = sample

    img_tensor = img_tensor.to(device).unsqueeze(0).float()
    centroid_coords = centroid_coords.to(device).float()

    output = model(img_tensor).squeeze(0)
    pred_dlat, pred_dlong = output[0].item(), output[1].item()
    pred_coords = (centroid_coords[0].item() + pred_dlat, centroid_coords[1].item() + pred_dlong)

    print(f"Inferred coordinates for index {args.idx}: ({pred_coords[0]:.6f}, {pred_coords[1]:.6f})")
    print(f"Reference coordinates: ({ref_coords[0].item():.6f}, {ref_coords[1].item():.6f})")
