SERVER_DIR=/Users/ofotech_fitri/Documents/fitri_github/tensorflow_object_detection_local
MODEL="faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8"
MODEL_DIR="output"
PIPELINE_CONFIG_PATH="training/$MODEL/pipeline.config"
CHECKPOINT_DIR=${MODEL_DIR}

python $SERVER_DIR/script/object_detection/model_main_tf2.py \
  --model_dir=$MODEL_DIR \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --checkpoint_dir=$CHECKPOINT_DIR
