import os
import h5py
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import SimpleITK as sitk
# List of directories to process
directories = [
    'brainTumorDataPublic_1-766',
    # 'brainTumorDataPublic_767-1532',
    # 'brainTumorDataPublic_1533-2298',
    # 'brainTumorDataPublic_2299-3064'
]

y_labels = []
filename_column = []

for directory in directories:

    # Set the paths for the Mat and Jpg folders
    mat_folder = f'./matfiles/{directory}'
    jpg_folder = f'./Jpg_images'

    # Create the Jpg folder if it doesn't exist
    if not os.path.exists(jpg_folder):
        os.makedirs(jpg_folder)

    # Iterate through files in the Mat folder
    for filename in os.listdir(mat_folder):
        # Construct the full file paths
        mat_filepath = os.path.join(mat_folder, filename)
        jpg_filepath = os.path.join(jpg_folder, filename.split(".")[0] + '.png')

        # Open the mat file
        with h5py.File(mat_filepath, 'r+') as f:
            cjdata = f['cjdata']
            image = np.array(cjdata.get('image')).astype(np.float64)
            label = cjdata.get('label')[0, 0]
            y_labels.append(label)
            filename_column.append(filename.split(".")[0])
            print(image)
            # Perform image processing
            hi = np.max(image)
            lo = np.min(image)
            print(hi)
            # image = (((image - lo) / (hi - lo)) * 255).astype(np.uint8)
            image = (((image - lo) / (hi - lo)) * 255)
            # cv2.imwrite(jpg_filepath, image)
            hi = np.max(image)
            print(hi)
            # t1_image = sitk.GetImageFromArray(image)

            # sitk.WriteImage(t1_image, jpg_filepath)
        break
    
df = pd.DataFrame({'filename':filename_column, 'label':y_labels})
df.to_csv('final_df.csv', index=False)