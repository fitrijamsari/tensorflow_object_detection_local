MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images
NO_LABEL_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/xml_error

python3 $SERVER_DIR/script/dataset_tools/check_xml_files.py -i $DATASET_DIR -o $NO_LABEL_DIR
