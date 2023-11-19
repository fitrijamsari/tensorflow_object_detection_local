"""
This script will resize images over max_size and update their corresponding XML files, if the images has been labelled.
"""

import os
import xml.etree.ElementTree as ET

from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")
parser.add_argument("-s", "--max_size", default = "1920", required=True, help="Maximum size to be scalled")

args = parser.parse_args()

# Path to the directory containing images and XML files
data_dir = args.input_dir
# data_dir = "./images"

# Desired maximum size for width or height
max_size = int(args.max_size)
# max_size = 1920

def resize_images_and_update_xml(data_dir, max_size):
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".JPG"):
                image_path = os.path.join(subdir, file)
                xml_path = os.path.join(subdir, file.split(".")[0] + ".xml")

                # Open and resize the image
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width > max_size or height > max_size:
                        if width >= height:
                            new_width = max_size
                            scale_factor = new_width / float(width)
                            new_height = int(height * scale_factor)
                        else:
                            new_height = max_size
                            scale_factor = new_height / float(height)
                            new_width = int(width * scale_factor)

                        resized_img = img.resize(
                            (new_width, new_height), Image.BILINEAR
                        )
                        resized_img.save(image_path)
                        print(f"SUCCESS: {image_path} has been resized")

                        # Update XML with new bounding box coordinates
                        tree = ET.parse(xml_path)
                        root = tree.getroot()

                        for size in root.iter("size"):
                            size.find("width").text = str(new_width)
                            size.find("height").text = str(new_height)

                        for obj in root.iter("object"):
                            for bbox in obj.iter("bndbox"):
                                xmin = int(bbox.find("xmin").text)
                                xmax = int(bbox.find("xmax").text)
                                ymin = int(bbox.find("ymin").text)
                                ymax = int(bbox.find("ymax").text)

                                bbox.find("xmin").text = str(int(xmin * scale_factor))
                                bbox.find("xmax").text = str(int(xmax * scale_factor))
                                bbox.find("ymin").text = str(int(ymin * scale_factor))
                                bbox.find("ymax").text = str(int(ymax * scale_factor))

                        tree.write(xml_path)
                        print(f"SUCCESS: {xml_path} bbox coordinate has been rescaled")


# Call the function
resize_images_and_update_xml(data_dir, max_size)
