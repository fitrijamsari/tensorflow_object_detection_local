MODEL_DIR=model_name
SERVER_DIR=/Users/ofotech_fitri/Documents/fitri_github/tensorflow_object_detection_local
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images
CSV_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/

python3 $SERVER_DIR/script/dataset_tools/xml_to_csv.py -i $DATASET_DIR -c $CSV_DIR