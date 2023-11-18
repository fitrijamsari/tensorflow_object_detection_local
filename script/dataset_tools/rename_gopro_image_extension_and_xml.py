import os
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")

args = parser.parse_args()

# Provide the root folder where you want to start the renaming process
input_directory = args.input_dir
# input_directory = './images'

def rename_files(root_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.JPG'):
                old_file_path = os.path.join(foldername, filename)
                new_name = os.path.splitext(filename)[0] + '.jpg'
                new_file_path = os.path.join(foldername, new_name)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {filename} to {new_name}")

                # Update XML files referencing the image filename
                update_xml_filenames(foldername, filename, new_name)

def update_xml_filenames(foldername, old_filename, new_filename):
    for filename in os.listdir(foldername):
        if filename.lower().endswith('.xml'):
            xml_file_path = os.path.join(foldername, filename)
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            for elem in root.iter('filename'):
                if elem.text == old_filename:
                    elem.text = new_filename

            tree.write(xml_file_path)

rename_files(input_directory)