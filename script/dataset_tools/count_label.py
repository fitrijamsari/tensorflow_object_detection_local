import xml.etree.ElementTree as ET
import glob
from collections import Counter
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_dir", dest = "input_dir", default = "model", required=True, help="Name of the dataset directory")

args = parser.parse_args()

# DATASET_DIR = '/Users/ofotech_fitri/Documents/ofo_dev_project/selia/bus_stop_m2_checked'
DATASET_DIR = args.input_dir
XML_PATHS = glob.glob(f'{DATASET_DIR}/**/*.xml', recursive=True)

existing_label_list=[]

def listXMLLabel(xml_file):
    xmlTree = ET.parse(xml_file)
    rootElement = xmlTree.getroot()
    for element in rootElement.findall('object'):
      label_name = element.find('name').text
      existing_label_list.append(label_name)
    return

def main():
  for xml_file in XML_PATHS:
      listXMLLabel(xml_file)
      
  total_label_count = Counter(existing_label_list)   
  print(total_label_count)

if __name__ == '__main__':
  main()

# python3 count_label.py -i /Users/ofotech_fitri/Documents/ofo_dev_project/selia/bus_stop_m2_checked

'''def count_label():
    list_total_label_count = [[x,label_classes.count(x)] for x in set(label_classes)]
    dict_total_label_count = dict((x,label_classes.count(x)) for x in set(label_classes))
    print(total_label_count)
    return'''

