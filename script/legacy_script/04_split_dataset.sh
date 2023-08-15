MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/temp/
OUTPUT_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/
RATIO=0.8

python3 $SERVER_DIR/script/dataset_tools/partition_dataset.py -x -i $DATASET_DIR -o $OUTPUT_DIR -r $RATIO