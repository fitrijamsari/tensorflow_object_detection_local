'''
This script will check if there are any label coordinate larger then the image size. 
If there is, we need to remove the files since it will disturb training job.
'''
import csv
import cv2 
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")

args = parser.parse_args()

# Path to the directory containing images and train.csv and test.csv files
csv_dir = args.input_dir
csv_files = [{csv_dir} +'/train.csv', {csv_dir} + 'test.csv']

for csv_file in csv_files:
    with open(csv_file, 'r') as fid:
        
        print('Checking file:', csv_file)
        
        file = csv.reader(fid, delimiter=',')
        first = True
        
        cnt = 0
        error_cnt = 0
        error = False
        for row in file:
            if error == True:
                error_cnt += 1
                error = False
                
            if first == True:
                first = False
                continue
            
            cnt += 1
            
            img_path, width, height, xmin, ymin, xmax, ymax = row[0], int(row[1]), int(row[2]), int(row[4]), int(row[5]), int(row[6]), int(row[7])
            
            path = img_path
            img = cv2.imread(path)
            
            
            if type(img) == type(None):
                error = True
                print('Could not read image', img)
                continue
            
            org_height, org_width = img.shape[:2]
            
            if org_width != width:
                error = True
                print('Width mismatch for image: ', img_path, width, '!=', org_width)
            
            if org_height != height:
                error = True
                print('Height mismatch for image: ', img_path, height, '!=', org_height)
            
            if xmin > org_width:
                error = True
                print('XMIN > org_width for file', img_path)
                
            if xmin < 0:
                error = True
                print('XMIN < 0 for file', img_path)
                
            if xmax > org_width:
                error = True
                print('XMAX > org_width for file', img_path)
            
            if ymin > org_height:
                error = True
                print('YMIN > org_height for file', img_path)
            
            if ymin < 0:
                error = True
                print('YMIN < 0 for file', img_path)
            
            if ymax > org_height:
                error = True
                print('YMAX > org_height for file', img_path)
            
            if xmin > xmax:
                error = True
                print('xmin >= xmax for file', img_path)
                
            if ymin > ymax:
                error = True
                print('ymin >= ymax for file', img_path)
            
            if error == True:
                print('Error for file: %s' % img_path)
            
        print('Checked %d files and realized %d errors' % (cnt, error_cnt))