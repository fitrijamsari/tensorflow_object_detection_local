MODEL=model_name
THRESH=0.3
SERVER_DIR=/Users/ofotech_fitri/Documents/fitri_github/tensorflow_object_detection_local

python $SERVER_DIR/script/object_detection/inference_draw_savedmodel.py -d $MODEL -thresh $THRESH
