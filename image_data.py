import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
class image_data:
    def __init__(self, id, ref_lat, ref_long, centroid_lat, centroid_long, num_of_cells, cells_lat, cells_long, cells_rssi, size=(32,32)): 

        self.id = id  # image id
        self.ref_lat = ref_lat  # reference latitude
        self.ref_long = ref_long  # reference longitude
        self.centroid_lat = centroid_lat  # centroid latitude
        self.centroid_long = centroid_long  # centroid longitude
        self.num_of_cells = num_of_cells  # number of recorded cells
        self.cells_lat = cells_lat  # list of latitude of cells
        self.cells_long = cells_long  # list of longitude of cells
        self.cells_rssi = cells_rssi  # list of rssi value of cells

        self.width = size[0]  # image's width
        self.height = size[1]  # image's height
        self.x = [] # x coordinate on image of cells
        self.y = [] # y coordinate on image of cells
        self.ref_x = self.width/2 - 1 # x coordinate on image of reference position
        self.ref_y = self.height/2 - 1 # y coordinate on image of reference position
        self.dmax = 0  # dmax as mentioned in the manuscript
        self.dlat = self.ref_lat - self.centroid_lat 
        self.dlong = self.ref_long - self.centroid_long

        # Initialize the image
        self.img = np.ndarray(size)
        self.img[:] = -114
    

    # Get x, y coordinates on image of cells and reference position
    def update_xy(self):

        # Calculate dmax
        for lat in self.cells_lat:
            self.dmax = max(self.dmax, abs(self.centroid_lat - lat))
        for long in self.cells_long:
            self.dmax = max(self.dmax, abs(self.centroid_long - long))

        # Calculate x, y coordinates on image of cells, the formula can be found in the manuscript
        for i in range(self.num_of_cells):
            d_long = self.cells_long[i] - self.centroid_long
            d_lat = self.cells_lat[i] - self.centroid_lat

            x_coord = self.width/2 - 1
            y_coord = self.height/2 - 1

            if (self.dmax > 0):
                x_coord = max(0, int((d_long/(2 * self.dmax) + 0.5) * self.width) - 1)
                y_coord = max(0, int((d_lat/(2 * self.dmax) + 0.5) * self.width) - 1)
            self.x.append(x_coord)
            self.y.append(y_coord)

        # Calculate x, y coordinates on image of reference position
        if (self.dmax > 0):
            d_long = self.ref_long - self.centroid_long
            d_lat = self.ref_lat - self.centroid_lat
            self.ref_x = max(0, int((d_long/(2 * self.dmax) + 0.5) * self.width) - 1)
            self.ref_y = max(0, int((d_lat/(2 * self.dmax) + 0.5) * self.width) - 1)


    '''
        Calculate the spreading model of a cell base on: Its rssi, its position and reference position -> spreading loss coefficient

        The spreading model of mobile signal can be modeled by the log distance path loss model
        https://en.wikipedia.org/wiki/Log-distance_path_loss_model

        Simplified: 
        RSSI(d) = RSSI(d0) + coeff * log10(d0/d)
        In the function: d0 = 1, RSSI(1) = 0 (at source)
    '''
    def cell_spread_model(self, cell_id):

        rssi = self.cells_rssi[cell_id]  # The cell's rssi
        cell_x = int(self.x[cell_id])  # The cell's x coordinate on image
        cell_y = int(self.y[cell_id])  # The cell's y coordinate on image
        d = float(np.sqrt((self.ref_x - cell_x) ** 2 + (self.ref_y - cell_y) ** 2))  # Distance between the cell and the reference position
        
        if d == 0:
            self.img[cell_x][cell_y] = 0
    
            return
        coeff = rssi/(np.log10(1/d) + 1e-8) 

        for x in range (0, self.width):
            for y in range (0, self.height):
                if (x == cell_x and y == cell_y):
                    continue
                dis = float(np.sqrt((x - cell_x) ** 2 + (y - cell_y) ** 2))
                self.img[x][y] = max(self.img[x][y], rssi + coeff *  np.log10(d/dis))
        self.img[cell_x][cell_y] = 0
        
        
    # Generate and return the image

    def gen_image(self) -> np.ndarray:
        self.update_xy()
        cnt = 2
        for cell_id in range(self.num_of_cells):
            self.cell_spread_model(cell_id)
            
        return self.img

class UrbanDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file, header=None)
        self.image_datas = []
        for index, row in tqdm(data.iterrows()):
            id = row[0]
            ref_lat = row[1]
            ref_long = row[2]
            num_of_cells = int(row[4])
            cells_lat = []
            cells_long = []
            cells_rssi = []
            for i in range(num_of_cells):
                cells_lat.append(row[i * 5 + 7])
                cells_long.append(row[i * 5 + 8])
                cells_rssi.append(row[i * 5 + 9])
            centroid_lat = sum(cells_lat) / num_of_cells
            centroid_long = sum(cells_long) / num_of_cells
            self.image_datas.append(image_data(id, ref_lat, ref_long, centroid_lat, centroid_long, num_of_cells, cells_lat, cells_long, cells_rssi))
    def __len__(self):
        return len(self.image_datas)
    def __getitem__(self, idx):
        image_instance = self.image_datas[idx]
        img_tensor = torch.from_numpy(image_instance.gen_image()).unsqueeze(0) #(1,size,size)
        img_tensor = img_tensor.float()
        ref_coords = torch.tensor([image_instance.ref_lat, image_instance.ref_long], dtype=torch.float32)
        centroid_coords = torch.tensor([image_instance.centroid_lat, image_instance.centroid_long], dtype=torch.float32)
        return img_tensor, ref_coords, centroid_coords


    