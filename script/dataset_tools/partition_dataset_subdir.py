import os
import random
import shutil
import argparse

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def count_images(directory, image_extensions):
    image_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1
    return image_count

def split_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                base_filename, file_extension = os.path.splitext(file)
                source_image_path = os.path.join(root, file)
                source_xml_path = os.path.join(root, base_filename + ".xml")

                # Determine whether to put the file in the train or test set
                if random.uniform(0, 1) < split_ratio:
                    destination_dir = train_dir
                else:
                    destination_dir = test_dir

                # Recreate the subfolder structure in the destination directory
                relative_path = os.path.relpath(root, source_dir)
                destination_subdir = os.path.join(destination_dir, relative_path)
                os.makedirs(destination_subdir, exist_ok=True)

                # Copy the image and its corresponding XML file
                destination_image_path = os.path.join(destination_subdir, file)
                destination_xml_path = os.path.join(destination_subdir, base_filename + ".xml")
                shutil.copy(source_image_path, destination_image_path)
                shutil.copy(source_xml_path, destination_xml_path)

    train_image_count = count_images(train_dir, image_extensions)
    test_image_count = count_images(test_dir, image_extensions)


    print("--------------------------DATASET SPLIT COMPLETED------------------------------")
    print(f'Train Dataset Directory: {train_dir}')
    print(f'Train Dataset Count: {train_image_count} Images')
    print(f'Test Dataset Directory: {test_dir}')
    print(f'Test Dataset Count: {test_image_count} Images')
    print("-------------------------------------------------------------------------------")

def main():

    # source_directory = "/Users/ofotech_fitri/Documents/tf_object_detection/script/script_temp/dataset/temp"
    # train_directory = "/Users/ofotech_fitri/Documents/tf_object_detection/script/script_temp/dataset/train"
    # test_directory = "/Users/ofotech_fitri/Documents/tf_object_detection/script/script_temp/dataset/test"
    # split_ratio = 0.8  # 80% for training, 20% for testing

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
        '-o', '--trainDir',
        dest = "trainDir",
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-t', '--testDir',
        dest = "testDir",
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
    
    args = parser.parse_args()

    os.makedirs(args.trainDir, exist_ok=True)
    os.makedirs(args.testDir, exist_ok=True)

    # split_dataset(source_directory, train_directory, test_directory, split_ratio)
    split_dataset(args.imageDir, args.trainDir, args.testDir, args.ratio)


if __name__ == '__main__':
    main()