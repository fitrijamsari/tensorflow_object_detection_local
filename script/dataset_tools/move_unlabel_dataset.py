# DESCRIPTION: THIS SCRIPT WILL MOVE: 1. IMAGES WITH NOT XML FILE 2. IMAGES WITH XML THAT DOES NOT CONTAIN LABEL (OBJECT)
import xml.etree.ElementTree as ET
import os
import glob
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")

args = parser.parse_args()

# DATASET_DIR = '/Users/ofotech_fitri/Documents/ofo_dev_project/selia/temp'
DATASET_DIR = args.input_dir
XML_PATHS = glob.glob(f'{DATASET_DIR}/**/*.xml', recursive=True)
IMG_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG']
IMAGE_PATHS =[]
[IMAGE_PATHS.extend(glob.glob(f'{DATASET_DIR}/**/'+ x, recursive=True)) for x in IMG_EXTS]
# IMAGE_PATHS = glob.glob(f'{DATASET_DIR}/**/*.jpg', recursive=True)

def checkImageWithXML(image_path):
    currentDir, imageFile = os.path.split(image_path)
    # currentDir = os.path.dirname(image_path)
    # imageFile = os.path.basename(image_path)
    imageFilenameOnly = imageFile.split(".")[0]
    if os.path.isfile(f'{currentDir}/{imageFilenameOnly}.xml'):
        # print(f'{image_path}: xml exist')
        return True
    else:
        print(f'{image_path}: xml do not exist.')
        return False
    
def moveImageIntoFolder(image_path):
    currentDir = os.path.dirname(image_path)
    noLabelFolder = createFolder(currentDir,folderName="no_label")
    shutil.move(image_path, noLabelFolder)
    return

def checkXMLLabelExist(xml_file):
    xmlTree = ET.parse(xml_file)
    rootElement = xmlTree.getroot()
    if len(rootElement.findall("object")) > 0:
        # print(f"{xml_file}: label exist") 
        return True
    else:
        print(f"{xml_file}: label do not exist") 
        return False
           
def moveImageXmlIntoFolder(xml_file):
    currentDir = os.path.dirname(xml_file)
    noLabelFolder = createFolder(currentDir,folderName="no_label")
    xmlFilenameOnly = xml_file.split('/')[-1].strip('.xml')
    for images in IMAGE_PATHS:
        imageFile = images.split("/")[-1]
        imageFilenameOnly = imageFile.split(".")[0]
        if imageFilenameOnly == xmlFilenameOnly:
            shutil.move(images, noLabelFolder)
            shutil.move(xml_file, noLabelFolder)
    return

def createFolder(outputDir,folderName):
    newDirectoryPath = os.path.join(outputDir, folderName)
    if not os.path.exists(newDirectoryPath):
        os.mkdir(newDirectoryPath)
    return newDirectoryPath

def main():
    if os.path.isdir(DATASET_DIR):
        #move images without xml first in a folder, then compare the remaining xml files, if label exist
        for image_path in IMAGE_PATHS:
            is_imageWithXML = checkImageWithXML(image_path)
            if is_imageWithXML == False:
                moveImageIntoFolder(image_path)

        for xml_file in XML_PATHS:
            is_XMLLabelExist = checkXMLLabelExist(xml_file)
            if is_XMLLabelExist == False:
                moveImageXmlIntoFolder(xml_file)

        print("---------COMPLETE REMOVING UNLABEL DATASET---------") 

    else:
        print("Error: Directory does not exist") 

if __name__ == '__main__':
  main()

# python3 remove_unlabel_dataset.py -i /Users/ofotech_fitri/Documents/ofo_dev_project/selia/bus_stop_m2_checked
