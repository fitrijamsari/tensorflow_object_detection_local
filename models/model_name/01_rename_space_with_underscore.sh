MODEL_DIR=model_name
SERVER_DIR=/Users/ofotech_fitri/Documents/fitri_github/tensorflow_object_detection_local
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images

python3 $SERVER_DIR/script/dataset_tools/clean_filename.py -i $DATASET_DIR