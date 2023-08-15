import xml.etree.ElementTree as ET
import os
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help = "Name of the dataset directory")

args = parser.parse_args()

#ASSIGN FOLDER PATH
DATASET_DIR = args.input_dir
XML_PATHS = glob.glob(f'{DATASET_DIR}/**/*.xml', recursive=True)
IMG_EXTS = ['*.jpg', '*.jpeg', '*.png']
IMAGE_PATHS = []
[IMAGE_PATHS.extend(glob.glob(f'{DATASET_DIR}/**/' + x, recursive=True)) for x in IMG_EXTS] 
#path = "/home/ofotech-ai2/Desktop/script_tool/sample dataset/one


#Check Image With XML
def checkImageWithXML(image_path):
    currentDir, imageFile = os.path.split(image_path)
    imageFilenameOnly = imageFile.split(".")[0]
    if os.path.isfile(f'{currentDir}/{imageFilenameOnly}.xml'):
        return True
    else:
        print(f'{image_path} : XML do not exist')
        return False


#Check Object Label on XML FIle
def checkXMLLabelExist(xmlFile):
    xmlTree = ET.parse(xmlFile)
    root = xmlTree.getroot()
    if len(root.findall("object")) > 0 :
        #print(f"{xmlFile}: Label exist")
        return True
    else:  
        print(f'{xmlFile}: Label do not exist')
        return False

#DeleteXML That has No Label
def DeleteImageXMLFile(xmlFile):
    xmlFileNameOnly = xmlFile.split('/')[-1].strip('.xml')
    for images in IMAGE_PATHS:
        imageFile = images.split("/")[-1]
        imageFileNameOnly  = imageFile.split(".")[0]
        if imageFileNameOnly == xmlFileNameOnly:
            os.remove(images)
            os.remove(xmlFile)
    return

def main():
    if os.path.isdir(DATASET_DIR):
        for image_path in IMAGE_PATHS:
            is_ImageWithXML = checkImageWithXML(image_path)
            if is_ImageWithXML == False:
                DeleteImageXMLFile(image_path)

        for xmlFIle in XML_PATHS:
            is_XMLLabelExist = checkXMLLabelExist(xmlFIle)
            if is_XMLLabelExist == False:
                DeleteImageXMLFile(xmlFIle)

        print("\n--------- DELETE UNLABEL DATASET COMPLETE ---------")
    
    else:
        print("Error: Directory does not exist")

if __name__ == '__main__':
  main()
                

#python3 remove_unlabel_dataset.py -i /home/ofotech-ai2/Desktop/script_tool/sample_dataset/one

#1 Assign Directory Path
#2 Read XML File and Image in Folder
#3 Detecting NonLabel Data
#4 Delete Nonlabel Dataset