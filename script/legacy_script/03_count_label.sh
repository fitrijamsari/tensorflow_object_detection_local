MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/
#DATASET_DIR=/media/irad/storage1/irad_dataset/arrow/

python3 $SERVER_DIR/script/dataset_tools/count_label.py -i $DATASET_DIR 
