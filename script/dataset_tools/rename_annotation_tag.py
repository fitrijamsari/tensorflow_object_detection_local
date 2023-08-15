import os
import xml.etree.ElementTree as ET
import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")

args = parser.parse_args()

DATASET_DIR = args.input_dir
XML_PATHS = glob.glob(f'{DATASET_DIR}/**/*.xml', recursive=True)

def renameXMLRootLabel(xml_file):
    xmlTree = ET.parse(xml_file)
    rootElement = xmlTree.getroot()
    print(f'{xml_file} -> Old Tag: {rootElement}')
    rootElement.tag = "annotation"
    print(f'{xml_file} -> New Tag: {rootElement}')
    print("----------------------------------------------")
    # xmlTree.write(xml_file,encoding='UTF-8',xml_declaration=True)
    xmlTree.write(xml_file)
    return

def main():
    if os.path.isdir(DATASET_DIR):
        for xml_file in XML_PATHS:
            renameXMLRootLabel(xml_file)
        print("------------RENAMING XML ROOT TAG TO ANNOTATION COMPLETED-------------")
    else: 
        print("Error: Directory does not exist")
           
if __name__ == '__main__':
    main()
