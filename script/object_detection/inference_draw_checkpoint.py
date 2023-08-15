###############################INFERENCE USING SAVED MODEL########################################
import os
import glob
import cv2
import time
import numpy as np
from PIL import Image
import argparse
from six import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

import warnings
warnings.filterwarnings('ignore')           # Suppress Matplotlib warnings

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--modeldir", dest = "modeldir", default = "model", help="Name of the model directory")
parser.add_argument("-thresh", "--thresh", dest = "threshold", default = "0.3", help="Min threshold", type=float)
# parser.add_argument("-size", "--binsize",dest = "binsize", help="Size", type=int)

args = parser.parse_args()

##########################################SET DIRECTORY#########################################
# MODEL_DIR_NAME = 'feeder_pillar'                                   # change this!! directory name of the trained model
MODEL_DIR_NAME = args.modeldir
TRAINED_MODEL_DIR = '/media/ofotechjkr/storage01/workspace/tensorflow_OD/ofotech_train/models'   
MODEL_DATE = '20200711'
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8'
PATH_TO_MODEL_DIR = f'{TRAINED_MODEL_DIR}/{MODEL_DIR_NAME}/output/model'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
PATH_TO_LABELS = f'{TRAINED_MODEL_DIR}/{MODEL_DIR_NAME}/dataset/labelmap.pbtxt'

PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"

# FOR DEBUGGING/MODEL TESTING
# IMAGE_DIR = f'{TRAINED_MODEL_DIR}/{MODEL_DIR_NAME}/debug/dataset'
#IMAGE_DIR="/media/ofotechjkr/New Volume/IMAGE PROCESS/2022_PAHANG"
IMAGE_DIR = "/media/ofotechjkr/storage01/AHADD/dataset/LPT2SLOPE"
OUTPUT_DIR = f'{TRAINED_MODEL_DIR}/{MODEL_DIR_NAME}/debug/output'

IMG_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
IMAGE_PATHS =[]
[IMAGE_PATHS.extend(glob.glob(f'{IMAGE_DIR}/**/'+ x, recursive=True)) for x in IMG_EXTS]
# IMAGE_PATHS = glob.glob(f'{IMAGE_DIR}/**/*.jpg', recursive=True)

detection_thresh = args.threshold
processing_time = []
image_count = []

if not os.path.exists(OUTPUT_DIR):
      os.makedirs(OUTPUT_DIR)

############################################LOAD MODEL#########################################
print('Loading model... ', end='')
start_time = time.time()
#limit gpu (option 1)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

# limit gpu (option 2)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4024)])
  except RuntimeError as e:
    print(e)
    
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

##################################LOAD IMAGES & INFERENCE ##################################  
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
# detection_model = load_model(PATH_TO_SAVED_MODEL)

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

def draw_inference(image_path):
  image_np = load_image_into_numpy_array(image_path)
  # image_np= cv2.cvtColor(image_np.copy(),cv2.COLOR_BGR2RGB)
  

  input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
  

  detections = detect_fn(input_tensor) #

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
  detections['num_detections'] = num_detections

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  label_id_offset = 1
  image_np_with_detections = image_np.copy()

  #Visualization of the resuls of detection
  viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes']+label_id_offset,
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=5,
          min_score_thresh=detection_thresh,
          agnostic_mode=False,
          line_thickness=5)

  # ymin, xmin, ymax, xmax = box

  # print(f"bbox: {detections['detection_boxes']}")
  # print(f"score: {detections['detection_scores']}")
  
  #save image with detections (later buat as a function)  
  imgname = image_path.split('/')[-1]
  dest=f'{OUTPUT_DIR}/{imgname}'
  temp_arr = np.array(image_np_with_detections) 
  # Convert RGB to BGR & Copy
  img_detected = temp_arr[:, :, ::-1].copy() 
  cv2.imwrite(dest,img_detected)
  print(f'Save Detected Image To: {dest}')
        
  image_count.append(imgname)  

def main():
  for image_path in IMAGE_PATHS:
    start_time = time.time()
    print('Running inference for {}'.format(image_path) + "\n", end='')

    draw_inference(image_path) #

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

