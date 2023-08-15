# Purpose: This script will convert all dataset images into csv. The csv will be use for data visualization & analytics purpose.
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

# Initiate argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as XML_DIR.",
                    type=str, default=None)

parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default=None)

args = parser.parse_args()

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob('{}/**/*.xml'.format(path), recursive=True):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = xml_file.split("/")[-1]
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        #value = None
        object_exists = root.find("object")
        if object_exists is not None:
            for member in root.findall('object'):
                bndbox = member.find('bndbox')
                value = (filename,
                        width,
                        height,
                        member.find('name').text,
                        int(bndbox.find('xmin').text),
                        int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text),
                        int(bndbox.find('ymax').text),
                        )
                xml_list.append(value)

        else:
            value = (filename,
                    width,
                    height,
                    'null',
                    'null',
                    'null',
                    'null',
                    'null'
                    )
            xml_list.append(value)

        print('value is', value)
                
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    xml_df = xml_to_csv(args.image_dir)
    xml_df.to_csv ((f'{args.csv_path}/dataset_labels.csv'), index=None)
    print('Successfully converted xml to csv.')      

if __name__ == '__main__':
  main()