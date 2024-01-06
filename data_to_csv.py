import os
import numpy as np
import pandas as pd
from PIL import Image

# Function to convert image to numpy array
def image_to_array(image_path):
    with Image.open(image_path) as img:
        return np.array(img)

# Paths to the folders containing the images
path_no_tumor = "/Users/manuelblanco/code/MBlancoC/Brain_Check/no"
path_yes_tumor = "/Users/manuelblanco/code/MBlancoC/Brain_Check/yes"

# List to hold the numpy arrays of images and labels
data = []

# Adding images from the 'no tumor' folder
for image_name in os.listdir(path_no_tumor):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Checking for image files
        image_path = os.path.join(path_no_tumor, image_name)
        data.append([image_to_array(image_path).tolist(), 'no'])  # Convert array to list for DataFrame

# Adding images from the 'yes tumor' folder
for image_name in os.listdir(path_yes_tumor):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Checking for image files
        image_path = os.path.join(path_yes_tumor, image_name)
        data.append([image_to_array(image_path).tolist(), 'yes'])  # Convert array to list for DataFrame

# Creating a DataFrame and saving it as a CSV
df = pd.DataFrame(data, columns=['image_array', 'label'])
csv_path = 'brain_tumor_dataset.csv'  # Specify your desired output path
df.to_csv(csv_path, index=False)
