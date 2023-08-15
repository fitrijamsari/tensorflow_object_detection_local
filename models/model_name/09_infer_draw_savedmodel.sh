MODEL=model_name
THRESH=0.3
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection

python $SERVER_DIR/script/object_detection/inference_draw_savedmodel.py -d $MODEL -thresh $THRESH
