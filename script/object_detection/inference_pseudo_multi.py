###############################INFERENCE USING SAVED MODEL WITH MULTI DETECTION AND AUTOLABEL########################################
import os
import glob
import cv2
import shutil
import script.legacy_script.xml_gen as xml_gen
import time
import numpy as np
from PIL import Image
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import warnings
warnings.filterwarnings('ignore')           # Suppress Matplotlib warnings

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--modeldir", dest = "modeldir", default = "model", help="Name of the model directory")
parser.add_argument("-thresh", "--thresh", dest = "threshold", default = "0.3", help="Min threshold", type=float)
# parser.add_argument("-size", "--binsize",dest = "binsize", help="Size", type=int)

args = parser.parse_args()

##########################################SET DIRECTORY#########################################
# MODEL_DIR_NAME = 'surface_crack'                                   # change this!! directory name of the trained model
MODEL_DIR_NAME = args.modeldir                                  
TRAINED_MODEL_DIR = '/media/ofotechjkr/storage01/2023_08_irad2/ml_training/models'   
MODEL_DATE = '20200711'
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8'
PATH_TO_MODEL_DIR = f'{TRAINED_MODEL_DIR}/{MODEL_DIR_NAME}/output/model'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
PATH_TO_LABELS = f'{TRAINED_MODEL_DIR}/{MODEL_DIR_NAME}/dataset/labelmap.pbtxt'

# FOR DEBUGGING/MODEL TESTING
IMAGE_DIR = f'{TRAINED_MODEL_DIR}/{MODEL_DIR_NAME}/pseudo/dataset'
# IMAGE_DIR = '/media/ofotechjkr/New Volume/IMAGE PROCESS/06_NEGERI_SEMBILAN_2022'
#/media/ofotechjkr/New Volume/dataset/driver
OUTPUT_DIR = f'{TRAINED_MODEL_DIR}/{MODEL_DIR_NAME}/pseudo/output'
NO_DETECTION_DIR = f'{TRAINED_MODEL_DIR}/{MODEL_DIR_NAME}/pseudo/output/no_detection'

IMG_EXTS = ['*.jpg', '*.jpeg', '*.png']
IMAGE_PATHS =[]
# NEW_IMAGE_PATHS = []
[IMAGE_PATHS.extend(glob.glob(f'{IMAGE_DIR}/**/'+ x, recursive=True)) for x in IMG_EXTS]

# for img in IMAGE_PATHS:
#   indicator = img.split("/")[-3]
#   if indicator == "PHOTO ASSET":
#     NEW_IMAGE_PATHS.append(img)


# IMAGE_PATHS = glob.glob(f'{IMAGE_DIR}/**/*.jpg', recursive=True)

detection_thresh = args.threshold
processing_time = []
image_count = []

############################################LOAD MODEL#########################################
def load_model(saved_model):
  start_time = time.time()
  
  #set gpu limit
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4024)])
    except RuntimeError as e:
      print(e)
      
  # Load saved model and build the detection function
  detection_model = tf.saved_model.load(saved_model)
    
  end_time = time.time()
  elapsed_time = end_time - start_time
  print('Loading Model: {} seconds'.format(elapsed_time))

  return detection_model
  
##################################LOAD IMAGES & INFERENCE ##################################  
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
detection_model = load_model(PATH_TO_SAVED_MODEL)

print('Start Image Loading & Inference......')

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def infer(model, image_file):
  image_np = load_image_into_numpy_array(image_file)

  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image_np)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis, ...]

  # input_tensor = np.expand_dims(image_np, 0)
  # detections = detect_fn(input_tensor)
  detections = model(input_tensor)

  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
  detections['num_detections'] = num_detections

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  return detections

def copyImageIntoClassesFolder(targetName, imageOutputDir, image_file):
        imageSaveDir = os.path.join(imageOutputDir, targetName)
        if not os.path.exists(imageSaveDir):
            os.mkdir(imageSaveDir)
        imageNameOnly = image_file.split("/")[-1].replace(" ", "_")
        imageOutputFilename = os.path.join(imageSaveDir, imageNameOnly)
        if not os.path.exists(imageOutputFilename):
            shutil.copy(image_file, imageOutputFilename)
        return imageOutputFilename

def xml_inference_multi(model, image_file):
    draw_and_replace = False
    out_detections = []
    
    img = cv2.imread(image_file)
    h, w, c = img.shape 
    if c == 1:
      img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
    else:
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb = img
   
    detections = infer(model, image_file)

    detected_scores = detections['detection_scores']
    detected_boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    detected_labels = [category_index[idx]['name'] for idx in classes]

    objects = []
    bounds = []
    voc_labels = {}
    voc_labels['all'] = []
    
    for box, label, score in zip(detected_boxes, detected_labels, detected_scores):
      if score >= detection_thresh:
        ymin, xmin, ymax, xmax = box
        xmin = int(round(xmin * w))
        ymin = int(round(ymin * h))
        xmax = int(round(xmax * w))
        ymax = int(round(ymax * h))
        boxes = [xmin, ymin, xmax, ymax]

        if xmin != xmax and ymin != ymax:
          if draw_and_replace:
            image_with_bbox = cv2.rectangle(img_rgb, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0,0,255), 2)
            cv2.imwrite(image_file, image_with_bbox)          

          objects.append(label)
          bounds.append(boxes)
          if label in voc_labels:
            voc_labels["labels"].append(label)
            voc_labels[label].append([label, xmin, ymin, xmax, ymax])
          else:
            voc_labels[label] = []
            voc_labels["labels"] = []
            voc_labels[label].append([label, xmin, ymin, xmax, ymax])
            voc_labels["labels"].append(label)
          voc_labels["all"].append([label, xmin, ymin, xmax, ymax])

          print([label, xmin, ymin, xmax, ymax], score)

    detection = {}
    detection["object"] = objects
    detection["bounds"] = bounds
    out_detections.append(detection)
    image_count.append(image_file)
    
    if not os.path.exists(OUTPUT_DIR):
      os.makedirs(OUTPUT_DIR)
    # imageNameOnly = image_file.split("/")[-1].replace(" ", "_")
    # imageOutputFilename = os.path.join(OUTPUT_DIR, imageNameOnly)
    # if not objects:
    #   shutil.copy(image_file, imageOutputFilename)
    #   os.remove(image_file)
    
    if voc_labels["all"]:
      num_of_classes_found = len(voc_labels["labels"])
      if num_of_classes_found == 1:
      # save into individual folder
        object_name = voc_labels["labels"][0]
        imageOutputFilename = copyImageIntoClassesFolder(object_name, OUTPUT_DIR, image_file)
        xml_gen.writeIntoXml(imageOutputFilename, voc_labels[object_name])
      else:
        # save into mixed folder
        imageOutputFilename = copyImageIntoClassesFolder("mixed_label", OUTPUT_DIR, image_file)
        xml_gen.writeIntoXml(imageOutputFilename, voc_labels["all"])
    else:
      pass
        #imageOutputFilename = copyImageIntoClassesFolder("no_detection", OUTPUT_DIR, image_file)
        # if os.path.exists(image_file):
        #   os.remove(image_file)
        
    '''for obj in objects:
      imageSaveDir = os.path.join(OUTPUT_DIR, obj)
      if not os.path.exists(imageSaveDir):
         os.mkdir(imageSaveDir)
      imageNameOnly = image_file.split("/")[-1].replace(" ", "_")
      imageOutputFilename = os.path.join(imageSaveDir, imageNameOnly)
      if not os.path.exists(imageOutputFilename):
        shutil.copy(image_file, imageOutputFilename)
    if objects:
      xml_gen.writeIntoXml(imageOutputFilename, voc_labels)
    else:
      os.remove(image_file)'''

    return out_detections

def main():
  for image_file in IMAGE_PATHS:
    start_time = time.time()
    print('Running inference for {}'.format(image_file) + "\n", end='')

    xml_inference_multi(detection_model,image_file)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    processing_time.append(elapsed_time)
    print('Processing Time per Image: {} seconds'.format(elapsed_time))

  total_processing_time = sum(processing_time)
  total_processing_time = "{:0.2f}".format(total_processing_time)
  total_image = len(image_count)

  print('--------------------------INFERENCE SUMMARY--------------------------------')
  print(f'Total Processed Image: {total_image}')
  print(f'Total Proccessing Time: {total_processing_time} seconds')
  print('---------------------------------------------------------------------------')

if __name__ == '__main__':
  main()
