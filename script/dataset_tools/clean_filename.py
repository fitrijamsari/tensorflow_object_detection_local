# This script will remove any spaces and symbols from foldername and filename
import os
import re
import argparse
import xml.etree.ElementTree as ET

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

            # Update references in XML files
            if extension.lower() == '.xml':
                image_name = filename.split('.')[0] + '.jpg'
                update_xml_references(root, filename, image_name)
    
    print("FINISHED RENAMING AND CLEANING")

#edit xml <filename> with new name
def update_xml_references(root, xml_filename, image_name):
    xml_path = os.path.join(root, xml_filename)
    
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for filename_element in root.iter('filename'):
            filename_element.text = image_name

        tree.write(xml_path)

if __name__ == "__main__":
    # target_directory = "/media/ofotechjkr/storage01/2023_08_irad2/ml_training/script/dataset_tools/images"
    target_directory = args.input_dir
    clean_and_rename(target_directory)