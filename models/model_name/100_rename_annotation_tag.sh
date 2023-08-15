MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/temp
#DATASET_DIR=/home/irad/workspace/irad_train/models/arrow_m2/dataset/temp

python3 $SERVER_DIR/script/dataset_tools/rename_annotation_tag.py -i $DATASET_DIR 
