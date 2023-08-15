import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")

args = parser.parse_args()

# DATASET_DIR = '/Users/ofotech_fitri/Documents/ofo_dev_project/selia/temp'
DATASET_DIR = args.input_dir
IMG_EXTS = ['*.jpg', '*.jpeg', '*.png']
IMAGE_PATHS =[]
[IMAGE_PATHS.extend(glob.glob(f'{DATASET_DIR}/**/'+ x, recursive=True)) for x in IMG_EXTS]

image_stats = []

for image in IMAGE_PATHS:
    img_bgr = cv2.imread(image, cv2.IMREAD_COLOR) 
    print(image)
    mean, std = cv2.meanStdDev(img_bgr)
    # dataset_stats.append(np.array([mean[::-1] / 255, std[::-1] / 255]))
    # image_stats.append(np.array([mean[::-1], std[::-1]])) #all item in the array reversed BGR - RGB
    #BGR format
    image_stats.append(np.array([mean, std]))

average_image_stats = np.mean(image_stats, axis=0)
print(f'Dataset Mean Average BGR: {average_image_stats[0]}')
print(f'Dataset Stds BGR: {average_image_stats[1]}')