import os
import argparse
import xml.etree.ElementTree as ET

# parser = argparse.ArgumentParser()

# parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")

# args = parser.parse_args()

# Provide the root folder where you want to start the renaming process
# input_directory = args.input_dir
input_directory = '/media/ofotechjkr/storage01/2023_08_irad2/ml_training/models/2023_11_01_surface_crack/dataset/images'

def update_xml_filenames(root_folder):
    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.xml'):
                xml_file_path = os.path.join(foldername, filename)
                tree = ET.parse(xml_file_path)
                root = tree.getroot()
                
                for elem in root.iter('filename'):
                    if elem.text.endswith('.JPG'):
                        elem.text = elem.text.split('.')[0] + '.jpg'
                        print(f"Updated filenames in {filename}")

                tree.write(xml_file_path)                

update_xml_filenames(input_directory)
print("Finish")