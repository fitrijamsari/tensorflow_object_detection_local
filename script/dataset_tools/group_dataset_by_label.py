import argparse
import os
import shutil
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input_dir",
    dest="input_dir",
    default="images_raw",
    required=True,
    help="Name of the dataset directory",
)
parser.add_argument(
    "-o",
    "--output_dir",
    dest="output_dir",
    default="images",
    required=True,
    help="Name of the output directory",
)
args = parser.parse_args()

# Path to the original dataset directory containing images and XML files
# original_dataset_dir = "./images"
original_dataset_dir = args.input_dir

# Path to create a copy of the dataset
# copied_dataset_dir = "./copied_images"
copied_dataset_dir = args.output_dir

# Copy the entire original dataset directory to a new location
shutil.copytree(original_dataset_dir, copied_dataset_dir)

# Create folders for each unique class and a "mix" folder for images with multiple labels in the copied dataset
classes = set()


def extract_classes_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter("object"):
        class_name = obj.find("name").text
        classes.add(class_name)


# Identify unique classes from XML files in the copied dataset
for subdir, dirs, files in os.walk(copied_dataset_dir):
    for file in files:
        if file.endswith(".xml"):
            xml_path = os.path.join(subdir, file)
            extract_classes_from_xml(xml_path)

# Create folders for each class and the "mix" folder in the copied dataset
for class_name in classes:
    os.makedirs(os.path.join(copied_dataset_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(copied_dataset_dir, "mix"), exist_ok=True)

# Move images based on classes in the copied dataset
for subdir, dirs, files in os.walk(copied_dataset_dir):
    for file in files:
        if file.endswith(".xml"):
            xml_path = os.path.join(subdir, file)
            image_path = os.path.join(
                subdir, file.split(".")[0] + ".jpg"
            )  # Change the image extension if different

            extract_classes_from_xml(xml_path)
            classes_in_image = set()

            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.iter("object"):
                class_name = obj.find("name").text
                classes_in_image.add(class_name)

            if len(classes_in_image) > 1:
                destination_folder = "mix"
            else:
                destination_folder = classes_in_image.pop()

            shutil.move(
                xml_path, os.path.join(copied_dataset_dir, destination_folder, file)
            )
            shutil.move(
                image_path,
                os.path.join(
                    copied_dataset_dir, destination_folder, file.split(".")[0] + ".jpg"
                ),
            )
            print(f"SUCCESS move {image_path} to {destination_folder}")

# Delete empty original folders in the copied dataset
for subdir, dirs, files in os.walk(copied_dataset_dir, topdown=False):
    for folder in dirs:
        folder_path = os.path.join(subdir, folder)
        if not os.listdir(folder_path):
            os.rmdir(folder_path)


print("SUCCESS: Group images with xml by classname")
