import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")

args = parser.parse_args()

# Provide the root folder where you want to start the renaming process
input_directory = args.input_dir
# input_directory = './images'

def rename_files(root_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.JPG'):
                file_path = os.path.join(foldername, filename)
                new_name = os.path.splitext(file_path)[0] + '.jpg'
                os.rename(file_path, new_name)
                print(f"Renamed {filename} to {os.path.basename(new_name)}")

rename_files(input_directory)
