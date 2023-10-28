import argparse
import os
import shutil
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input_dir",
    dest="input_dir",
    default="model",
    required=True,
    help="Name of the dataset directory",
)
parser.add_argument(
    "-o",
    "--output_dir",
    dest="output_dir",
    default="no_label",
    required=True,
    help="Name of the on_label directory",
)

args = parser.parse_args()

# Define the directory where your dataset is stored
dataset_dir = args.input_dir

# Define the output directory for cleaned data
output_dir = args.output_dir

# Supported image file extensions
image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def has_labels(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return len(root.findall(".//object")) > 0
    except ET.ParseError:
        return False


def clean_dataset(dataset_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root_dir, _, files in os.walk(dataset_dir):
        for file in files:
            _, file_extension = os.path.splitext(file)
            if file_extension.lower() in image_extensions:
                image_path = os.path.join(root_dir, file)
                xml_path = os.path.splitext(image_path)[0] + ".xml"

                if os.path.exists(xml_path):
                    if not has_labels(xml_path):
                        # Move both image and XML with labels
                        print("Removed Image Contain Unlabelled Data")
                        shutil.move(
                            image_path, os.path.join(output_dir, file)
                        )
                        shutil.move(
                            xml_path,
                            os.path.join(
                                output_dir, os.path.basename(xml_path)
                            ),
                        )
                else:
                    # Move images without corresponding XML
                    print(
                        "Removed Image did not contain respective xml"
                        " file"
                    )
                    shutil.move(
                        image_path, os.path.join(output_dir, file)
                    )


if __name__ == "__main__":
    clean_dataset(dataset_dir, output_dir)
    print("Dataset cleaning complete.")
