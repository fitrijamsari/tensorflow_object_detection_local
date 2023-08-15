# DESCRIPTION: THIS SCRIPT WILL GROUP IMAGES BASED ON THEIR RESPECTIVE LABELS
# MAKE SURE ALL IMAGES HAVES THEIR LABELS ELSE RUN MOVE_NO_LABEL SCRIPT BEFORE RUNNING THIS SCRIPT
import glob
import xml.etree.ElementTree as ET
import time
import shutil
import os


# tree = ET.parse('country_data.xml')
# root = tree.getroot()


def main():
    IMAGE_DIR = "/home/ofotechjkr/workspace/tensorflow_OD/ofotech_train/models/07_feeder_pillar/dataset/test/"
    IMG_EXTS = ['*.jpg', '*.jpeg', '*.png']
    IMAGE_PATHS =[]
    
    XML_DIR = "/home/ofotechjkr/workspace/tensorflow_OD/ofotech_train/models/07_feeder_pillar/dataset/test/**/" + "*.xml"
    
    [IMAGE_PATHS.extend(glob.glob(f'{IMAGE_DIR}/**/'+ x, recursive=True)) for x in IMG_EXTS]
    XML_PATHS = glob.glob(XML_DIR,recursive=True )

    OUTPUT_DIR = "/home/ofotechjkr/workspace/tensorflow_OD/ofotech_train/models/07_feeder_pillar/dataset/output_grouped"

    # print(IMAGE_PATHS)
    # print(XML_PATHS)



    for image_file,xml_file in zip(sorted(IMAGE_PATHS),sorted(XML_PATHS)):
        xmlTree = ET.parse(xml_file)
        rootElement = xmlTree.getroot()
        object = rootElement.findall("object")

        for every_object in object:
            
            if every_object.find('name').text == "block_wall":
                imageNameOnly = image_file.split('/')[-1]
                xmlNameOnly = xml_file.split('/')[-1]
                print(imageNameOnly + " and " + xmlNameOnly) 
                imageOutputFilename = os.path.join(OUTPUT_DIR, imageNameOnly)
                shutil.copy(image_file, imageOutputFilename)
                xmlOutputFilename = os.path.join(OUTPUT_DIR, xmlNameOnly)
                shutil.copy(xml_file, xmlOutputFilename)
            else:
                pass
        
        




if __name__ == '__main__':
    main()