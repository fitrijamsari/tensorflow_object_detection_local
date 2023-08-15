from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import glob
import fnmatch

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']


def dict_to_tf_example(data,
                    img_path,
                    label_map_dict,
                    ignore_difficult_instances=False):

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

        difficult_obj.append(int(difficult))

        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map_dict[obj['name']])
        if label_map_dict[obj['name']] is None:
            print(label_map_dict[obj['name']])
        # truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        # 'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):

    #-------------------------adjustment for multiple image format------------------------
    data_dir = FLAGS.data_dir
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    IMG_EXTS = ['*.jpg', '*.jpeg', '*.png']
    IMAGE_PATHS = []
    [IMAGE_PATHS.extend(glob.glob(f'{data_dir}/**/' + x, recursive=True)) for x in IMG_EXTS] 

    for imgfile in IMAGE_PATHS:   
        print(f'Img: {imgfile}')
        xmlfile = imgfile.split('.')[0] + '.xml'
        with tf.gfile.GFile(xmlfile, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        # print(dataset_util.recursive_parse_xml_to_dict(xml))
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, imgfile, label_map_dict,
                                        FLAGS.ignore_difficult_instances)
        writer.write(tf_example.SerializeToString())

    # -------------------------------------------------------------------------------------    
    # data_dir = FLAGS.data_dir
    # writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    # label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    # sub_directory = glob.glob(f'{data_dir}/**/*.jpg', recursive=True)

    #  # for imgfile in glob.glob(os.path.join(data_dir,'*.jpg')):
    # for imgfile in sub_directory:   
    #     print(f'Img: {imgfile}')
    #     xmlfile = imgfile.replace('.jpg','.xml')
    #     with tf.gfile.GFile(xmlfile, 'r') as fid:
    #         xml_str = fid.read()
    #     xml = etree.fromstring(xml_str)
    #     # print(dataset_util.recursive_parse_xml_to_dict(xml))
    #     data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    #     tf_example = dict_to_tf_example(data, imgfile, label_map_dict,
    #                                     FLAGS.ignore_difficult_instances)
    #     writer.write(tf_example.SerializeToString())

    # writer.close()
    # print('Success create tf record.',flush=True)
   

    writer.close()
    print('Success create tf record.',flush=True)

if __name__ == '__main__':
    tf.app.run()
