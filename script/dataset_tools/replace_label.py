import argparse
import glob
import xml.etree.ElementTree as ET
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input_dir",
    dest="input_dir",
    default="model",
    required=True,
    help="Name of the dataset directory",
)

args = parser.parse_args()

DATASET_DIR = args.input_dir
XML_PATHS = glob.glob(f"{DATASET_DIR}/**/*.xml", recursive=True)

existing_label_list = []
new_label_list = []
# original_label ="bus_stop"
# new_label ="haha"


def changeXMLLabel(xml_file, original_label, new_label):
    xmlTree = ET.parse(xml_file)
    rootElement = xmlTree.getroot()
    for element in rootElement.findall("object"):
        label = element.find("name")
        if label.text == original_label:
            label.text = new_label
    # xmlTree.write(xml_file,encoding='UTF-8',xml_declaration=True)
    xmlTree.write(xml_file)
    return


def storeLabelXML(xml_file, store_list):
    xmlTree = ET.parse(xml_file)
    rootElement = xmlTree.getroot()
    for element in rootElement.findall("object"):
        label_name = element.find("name").text
        store_list.append(label_name)
    return


def main():
    for xml_file in XML_PATHS:
        storeLabelXML(xml_file, existing_label_list)
    total_existing_label_count = Counter(existing_label_list)
    print("-------------------EXISTING LABEL---------------------")
    print(f"Existing Label {total_existing_label_count} \n")

    print("---------------------USER INPUT-----------------------")
    select_label_to_change = input(
        "Please enter which label do you want to replace: "
    )

    if select_label_to_change in existing_label_list:
        select_new_label = input("Please enter new label: ")
        for xml_file in XML_PATHS:
            changeXMLLabel(
                xml_file, select_label_to_change, select_new_label
            )
            storeLabelXML(xml_file, new_label_list)

        total_new_label_count = Counter(new_label_list)
        print("-----------------SUMMARY-------------------")
        print(
            f"Success: '{select_label_to_change}' label has been"
            f" replace with '{select_new_label}'"
        )
        print(f"New Label {total_new_label_count}")
        print("-------------------------------------------")
    else:
        print(
            f"Error: Label '{select_label_to_change}' does not exist"
        )


if __name__ == "__main__":
    main()
