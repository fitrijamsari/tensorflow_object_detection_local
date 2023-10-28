MODEL_DIR=model_name
SERVER_DIR=/Users/ofotech_fitri/Documents/fitri_github/tensorflow_object_detection_local
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images
CORRUPT_IMAGE_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/corrupted_images

python3 $SERVER_DIR/script/dataset_tools/check_corrupted_images.py -i $DATASET_DIR -o $CORRUPT_IMAGE_DIR