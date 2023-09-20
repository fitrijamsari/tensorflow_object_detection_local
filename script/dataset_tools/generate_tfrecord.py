""" Sample TensorFlow XML-to-TFRecord converter
usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]
optional arguments:
  -h, --help            show this help message and exit
  -x XML_DIR, --xml_dir XML_DIR
                        Path to the folder where the input .xml files are stored.
  -l LABELS_PATH, --labels_path LABELS_PATH
                        Path to the labels (.pbtxt) file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output TFRecord (.record) file.
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as XML_DIR.
  -c CSV_PATH, --csv_path CSV_PATH
                        Path of output .csv file. If none provided, then no file will be written.
"""

import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
import PIL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image, UnidentifiedImageError
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter")
parser.add_argument("-x",
                    "--xml_dir",
                    help="Path to the folder where the input .xml files are stored.",
                    type=str)
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)
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

if args.image_dir is None:
    args.image_dir = args.xml_dir

label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)


def xml_to_csv(path):
    xml_list = []
    # for xml_file in glob.glob(path + '/*.xml'):
    for xml_file in glob.glob('{}/**/*.xml'.format(path), recursive=True):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # this is dangeraous since images can be in various extension!!!
        # make sure the filename is the path to the local directory and not the path in the xml files. xml path might be different since labeled can be done from different computer.
        # filename = xml_file.split('.')[0] + '.jpg'
        folder_path = os.path.dirname(xml_file)
        image_name = root.find('filename').text.split('/')[-1]
        filename = (f'{folder_path}/{image_name}')
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
                    '-1',
                    '-1',
                    '-1',
                    '-1',
                    '-1'
                    )
            xml_list.append(value)

        # print('value is', value)
                
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    print(xml_df)
    return xml_df


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    # try:
    #     image = Image.open(encoded_jpg_io)
    # except UnidentifiedImageError as e:
    #     print(f"Error: {e}")
    #     print(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        if int(row['xmin']) > -1:
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(class_text_to_int(row['class']))

        else:
            xmins.append(0)
            xmaxs.append(0)
            ymins.append(0)
            ymaxs.append(0)
            classes_text.append('something'.encode('utf8'))  # this does not matter for the background
            classes.append(5000)


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    for folder in ['train','test']:
        writer = tf.python_io.TFRecordWriter(f'{args.output_path}/{folder}.tfrecord')
        path = os.path.join(args.image_dir)
        examples = xml_to_csv(f'{args.xml_dir}/{folder}')
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecord file: {}'.format(f'{args.output_path}/{folder}.tfrecord'))
        if args.csv_path is not None:
            examples.to_csv((f'{args.csv_path}/{folder}.csv'), index=None)
            print('Successfully created the CSV file: {}'.format(f'{args.csv_path}/{folder}.csv'))


if __name__ == '__main__':
    tf.app.run()
