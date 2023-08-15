MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images
TRAIN_DIR=$SERVER_DIR/models/$MODEL_DIR/training/train
TEST_DIR=$SERVER_DIR/models/$MODEL_DIR/training/test
RATIO=0.8

python3 $SERVER_DIR/script/dataset_tools/partition_dataset_subdir.py -i $DATASET_DIR -o $TRAIN_DIR -t $TEST_DIR -r $RATIO