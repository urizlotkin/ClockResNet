import pandas as pd
import json
from PIL import Image
import rasterio
import numpy as np
import os
from tqdm import tqdm
import pickle

# Initialize a list to hold your data
all_data = []

# Define the base directory
base_dir = 'BigEarthNet-v1.0'
# Loop through each subfolder in the base directory

for folder_name in tqdm(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, folder_name)
    
    # Check if it's a directory
    if os.path.isdir(folder_path):
    #     # Find the JSON file and read the acquisition_date
        json_file_path = os.path.join(folder_path, folder_name + '_labels_metadata.json')
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Access the acquisition_date field
        acquisition_date = data['acquisition_date']
        
        # Loop through each TIFF file in the subfolder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.tif'):
                tif_file_path = os.path.join(folder_path, file_name)
                
                # Read the TIFF file and convert it to a matrix
                try:
                    with rasterio.open(tif_file_path) as src:
                        img_matrix = src.read()  # This reads all bands; adjust if you need something different
                        # print("Shape: ", img_matrix.shape)
                        shape = img_matrix.shape
                        if shape[1] == 120:
                            all_data.append({
                            'image_data': img_matrix,
                            'acquisition_date': acquisition_date
                            })
                except:
                    continue

                    # print("Matrix: ", img_matrix)
                    # Append the data to your list

# Convert your list of data into a pandas DataFrame
df = pd.DataFrame(all_data)   
# df = pd.DataFrame(data)


# Save the DataFrame as a pickle file
df.to_pickle('df_120.pkl')



