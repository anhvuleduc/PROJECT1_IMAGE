# Generate images base on RSSI and cell locations

## How to use
- Clone the repository
```
git clone https://github.com/anhvuleduc/PROJECT1_IMAGE.git
cd PROJECT1_IMAGE
```

- Install the dependencies
```
pip install -r requirements.txt
```

- Run mycode.ipynb (change the index in the SHOW IMAGE cell to show to corresponding data row's image)

## CSV file data format
- Data file is logFile_urban_data.csv
- Data format: Each row is: 
   - data_id, lat_ref,lon_ref, time, number of cells, (lac,cid,cell_lat,cell_lon,rssi), ...

## Path loss formula
- The attenuation of mobile signal can be modeled using the Log-Distance Path Loss Model:  ![Alttext](./PathLoss.png)

- Also, it cell can has its own surrounding environment, resulting in different path loss exponent (coefficient)

## 