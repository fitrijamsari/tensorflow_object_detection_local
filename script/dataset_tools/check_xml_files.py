import os
import shutil
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")
parser.add_argument("-o", "--output_dir", dest = "output_dir", default = "no_label", required=True, help="Name of the on_label directory")

args = parser.parse_args()

# Define the directory where your dataset is stored
# input_directory = "/media/ofotechjkr/storage01/2023_08_irad2/ml_training/models/2023_08_08_signboard/dataset/images"
input_directory = args.input_dir

# Define the output directory for cleaned data
# output_dir = "/media/ofotechjkr/storage01/2023_08_irad2/ml_training/models/2023_08_08_signboard/dataset/xml_error"
output_directory = args.output_dir

# Supported image file extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG','.PNG']

def move_xml_files_with_errors(directory, output_dir):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".xml"):
                xml_file_path = os.path.join(root, filename)
                try:
                    # Attempt to parse the XML file
                    ET.parse(xml_file_path)
                except ET.ParseError as e:
                    print(f"{xml_file_path}: Error - {e}")
                    # Move the XML file to the output directory
                    xml_output_path = os.path.join(output_dir, filename)
                    shutil.move(xml_file_path, xml_output_path)
                    # Move associated image files (if they exist)
                    base_filename, _ = os.path.splitext(filename)
                    for ext in image_extensions:
                        image_filename = base_filename + ext
                        image_file_path = os.path.join(root, image_filename)
                        if os.path.exists(image_file_path):
                            image_output_path = os.path.join(output_dir, image_filename)
                            shutil.move(image_file_path, image_output_path)

if __name__ == "__main__":
    # Replace 'your_directory_path' with the directory where your XML files are located
    # input_directory = "your_directory_path"
    # Replace 'output_dir' with the directory where you want to move the files with errors
    # output_directory = "output_dir"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    move_xml_files_with_errors(input_directory, output_directory)
    print("Complete XML Files Format Check: No XML Format Error")
