#!/usr/bin/python3
import os
import cv2
import xml.etree.cElementTree as ET


def writeIntoXml(image_filename, voc_labels):
    image = cv2.imread(image_filename)
    h, w, c = image.shape
    base, _ = os.path.split(image_filename)

    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_filename
    ET.SubElement(root, "folder").text = base
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])

    tree = ET.ElementTree(root)
    xmlOutputFilename = image_filename.split(".")[0] + ".xml"
    tree.write(xmlOutputFilename)
