import argparse
import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request
import numpy as np
import sys
from PIL import Image

parser = argparse.ArgumentParser(description='r/Fakeddit image downloader')

parser.add_argument('type', type=str, help='train, validate, or test')

args = parser.parse_args()

df = pd.read_csv(args.type, sep="\t", nrows=500)
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)


pbar = tqdm(total=len(df))

if not os.path.exists("images_test"):
  os.makedirs("images_test")

textWithImages = []
sentimentValues = []

image2row = 0


def is_corrupted_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.load()
        return False  # The image file is not corrupted
    except Exception as e:
        print(f"Error: {e}")
        return True  # The image file is corrupted
    
for index, row in df.iterrows():
  if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
    image_url = row["image_url"]
    try: 
        filepath = "images_test/" + str(image2row) + ".jpg"
        urllib.request.urlretrieve(image_url, filepath)

        if is_corrupted_image(filepath) == False:
            textWithImages.append(row["clean_title"])
            sentimentValues.append(row["2_way_label"])
            image2row += 1
        else:
           os.remove(filepath)
    except Exception as e:
      print(f"couldnt find image for row number : {index}")
    
  pbar.update(1)

data = {'text with images': textWithImages, '2_way_label' : sentimentValues}
dfWithImages = pd.DataFrame(data)
print("done")

dfWithImages.to_csv('final_df_test.csv', index=False)