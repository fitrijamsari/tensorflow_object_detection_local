MODEL_DIR=model_name
SERVER_DIR=/Users/ofotech_fitri/Documents/fitri_github/tensorflow_object_detection_local
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/training

python $SERVER_DIR/script/dataset_tools/generate_tfrecord.py -x $DATASET_DIR -l $DATASET_DIR/labelmap.pbtxt -o $DATASET_DIR -c $DATASET_DIR