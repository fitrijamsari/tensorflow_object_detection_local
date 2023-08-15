# This script will remove any spaces and symbols from foldername and filename
import os
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")

args = parser.parse_args()

def clean_and_rename(path):
    for root, dirs, files in os.walk(path):
        for dirname in dirs:
            new_dirname = re.sub(r'[^\w\s]', '', dirname)  # Remove symbols from directory name
            new_dirname = new_dirname.replace(' ', '_')  # Replace spaces with underscores
            original_path = os.path.join(root, dirname)
            new_path = os.path.join(root, new_dirname)
            
            if original_path != new_path:
                os.rename(original_path, new_path)
                print(f'Renamed Directory: {original_path} -> {new_path}')

        for filename in files:
            name, extension = os.path.splitext(filename)
            new_name = re.sub(r'[^\w\s]', '', name)  # Remove symbols from name
            new_name = new_name.replace(' ', '_')  # Replace spaces with underscores
            new_filename = new_name + extension
            original_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_filename)
            
            if original_path != new_path:
                os.rename(original_path, new_path)
                print(f'Renamed File: {original_path} -> {new_path}')
    
    print("FINISHED RENAMING AND CLEANING")

if __name__ == "__main__":
    # target_directory = "/media/ofotechjkr/storage01/tf_object_detection/script/dataset_tools/images"
    target_directory = args.input_dir
    clean_and_rename(target_directory)