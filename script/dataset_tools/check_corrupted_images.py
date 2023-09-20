import os
import shutil
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")
parser.add_argument("-o", "--output_dir", dest = "output_dir", default = "no_label", required=True, help="Name of the on_label directory")

args = parser.parse_args()

# Define the directory where your dataset is stored
# root_dir = "/media/ofotechjkr/storage01/2023_08_irad2/ml_training/models/2023_08_08_signboard/dataset/images"
root_dir = args.input_dir

# Define the output directory for cleaned data
# corrupted_images_dir = "/media/ofotechjkr/storage01/2023_08_irad2/ml_training/models/2023_08_08_signboard/dataset/corrupted_images"
corrupted_images_dir = args.output_dir

# Ensure the "corrupted_images" directory exists; create it if not
if not os.path.exists(corrupted_images_dir):
    os.makedirs(corrupted_images_dir)

# Function to check if an image is corrupted
def is_image_corrupted(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False  # Image is not corrupted
    except Exception:
        return True  # Image is corrupted

# Recursively traverse subdirectories
for root, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # Check if the image is corrupted
            image_path = os.path.join(root, filename)
            if is_image_corrupted(image_path):
                # Move the corrupted image and its XML file to the "corrupted_images" folder
                xml_filename = os.path.splitext(filename)[0] + ".xml"
                xml_path = os.path.join(root, xml_filename)

                # Create a subfolder structure in "corrupted_images" to mimic the original directory structure
                relative_path = os.path.relpath(root, root_dir)
                destination_folder = os.path.join(corrupted_images_dir, relative_path)
                os.makedirs(destination_folder, exist_ok=True)

                # Move the corrupted image
                shutil.move(image_path, os.path.join(destination_folder, filename))

                # If there is an XML file, move it as well
                if os.path.exists(xml_path):
                    shutil.move(xml_path, os.path.join(destination_folder, xml_filename))
                    
                print(f"Moved corrupted image: {image_path} and XML: {xml_path} to {destination_folder}")

print("Corrupted images check and move process complete.")