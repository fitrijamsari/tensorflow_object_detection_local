""" usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]

Partition dataset of images into training and testing sets

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        Path to the output folder where the train and test dirs should be created. Defaults to the same directory as IMAGEDIR.
  -r RATIO, --ratio RATIO
                        The ratio of the number of test images over the total number of images. The default is 0.1.
  -x, --xml             Set this flag if you want the xml annotation files to be processed and copied over.
"""
import os
import re
from shutil import copyfile, move
import argparse
import math
import random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#future changes: adjust so that it can split accordingly to subdirectory (optional)

def iterate_dir(source, dest, ratio, copy_xml):
    source = source.replace('\\', '/')
    dest = dest.replace('\\', '/')
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png|.JPG|.JPEG)$', f)]

    num_images = len(images)
    num_test_images = math.ceil(ratio*num_images)
    num_train_images = num_images - num_test_images

    for i in range(num_test_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        move(os.path.join(source, filename),
                 os.path.join(test_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            move(os.path.join(source, xml_filename),
                     os.path.join(test_dir,xml_filename))
        images.remove(images[idx])

    for filename in images:
        move(os.path.join(source, filename),
                 os.path.join(train_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            move(os.path.join(source, xml_filename),
                     os.path.join(train_dir, xml_filename))
    
    return num_train_images, num_test_images

def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        dest = "imageDir",
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default=os.getcwd()
    )
    parser.add_argument(
        '-o', '--outputDir',
        dest = "outputDir",
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-r', '--ratio',
        dest = "ratio",
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=0.1,
        type=float)
    parser.add_argument(
        '-x', '--xml',
        help='Set this flag if you want the xml annotation files to be processed and copied over.',
        action='store_true'
    )
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    # Now we are ready to start the iteration
    num_train_images, num_test_images = iterate_dir(args.imageDir, args.outputDir, args.ratio, args.xml)

    print("--------------------------DATASET SPLIT COMPLETED------------------------------")
    print(f'Dataset Split Directory: {args.outputDir}')
    print(f"Dataset Count: Train: {num_train_images}, Test: {num_test_images}")
    print("-------------------------------------------------------------------------------")

if __name__ == '__main__':
    main()


# For example
# python3 partition_dataset.py -x -i /home/irad/workspace/irad_train/models/arrow_single/dataset/temp/ -r 0.2

# Once the script has finished, two new folders should have been created under bus_stop/dataset, 
# namely dataset/train and dataset/test, containing 80% and 20% of the images (and *.xml files),respectively. 
