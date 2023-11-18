import argparse
import os
import shutil
from struct import unpack

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    dest="input_dir",
    default="input_dir",
    required=True,
    help="Name of the dataset directory",
)
parser.add_argument(
    "-o",
    "--output_dir",
    dest="output_dir",
    default="output_dir",
    required=True,
    help="Name of the on_label directory",
)
args = parser.parse_args()

marker_mapping = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC0: "Start of Frame",
    0xFFC4: "Define Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image",
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, "rb") as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while True:
            (marker,) = unpack(">H", data[0:2])
            if marker == 0xFFD8:
                data = data[2:]
            elif marker == 0xFFD9:
                return
            elif marker == 0xFFDA:
                data = data[-2:]
            else:
                (lenchunk,) = unpack(">H", data[2:4])
                data = data[2 + lenchunk :]
            if len(data) == 0:
                break


def is_image_corrupted(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False  # Image is not corrupted
    except Exception:
        return True  # Image is corrupted


def move_corrupted_images(root_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg")):
                image_path = os.path.join(root, filename)
                xml_filename = os.path.splitext(filename)[0] + ".xml"
                xml_path = os.path.join(root, xml_filename)

                # Check if the image is corrupted
                if is_image_corrupted(image_path):
                    # Move the corrupted image and its XML file to the output directory
                    output_image_path = os.path.join(output_dir, filename)
                    output_xml_path = os.path.join(output_dir, xml_filename)

                    shutil.move(image_path, output_image_path)
                    shutil.move(xml_path, output_xml_path)

                    print(
                        f"Moved corrupted image: {image_path} and XML: {xml_path} to"
                        f" {output_dir}"
                    )
                else:
                    # If the image is not corrupted, try decoding it
                    image = JPEG(image_path)
                    try:
                        image.decode()
                    except:
                        # If decoding fails, move the image and its XML file to the output directory
                        output_image_path = os.path.join(output_dir, filename)
                        output_xml_path = os.path.join(output_dir, xml_filename)

                        shutil.move(image_path, output_image_path)
                        shutil.move(xml_path, output_xml_path)

                        print(
                            f"Moved corrupted image: {image_path} and XML:"
                            f" {xml_path} to {output_dir}"
                        )


if __name__ == "__main__":
    move_corrupted_images(args.input_dir, args.output_dir)

    print("Corrupted images check and move process complete.")
