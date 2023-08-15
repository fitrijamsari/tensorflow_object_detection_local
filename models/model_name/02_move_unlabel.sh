MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images

python3 $SERVER_DIR/script/dataset_tools/move_unlabel_dataset.py -i $DATASET_DIR 
