MODEL_DIR=04_signboard_fasterrcnn_inception640
SERVER_DIR=/media/ofotechjkr/storage01/2023_08_irad2/ml_training
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset
OUTPUT_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset

python $SERVER_DIR/script/dataset_tools/generate_tfrecord.py -x $DATASET_DIR/train/ -l $DATASET_DIR/labelmap.pbtxt -o $DATASET_DIR/train.tfrecord -c $DATASET_DIR/train.csv