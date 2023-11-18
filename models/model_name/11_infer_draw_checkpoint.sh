MODEL=model_name
THRESH=0.3
SERVER_DIR=/media/ofotechjkr/storage01/2023_08_irad2/ml_training
TRAIN_DIR=$SERVER_DIR/models/$MODEL

python $SERVER_DIR/script/object_detection/inference_draw_checkpoint.py -t $TRAIN_DIR -thresh $THRESH