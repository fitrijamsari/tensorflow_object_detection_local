SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
MODEL="faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8"
CHECKPOINT_DIR="output"
PIPELINE_CONFIG_PATH="training/$MODEL/pipeline.config"
OUTPUT_DIR="${CHECKPOINT_DIR}/model"

python $SERVER_DIR/script/object_detection/exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_dir ${CHECKPOINT_DIR} \
    --output_directory ${OUTPUT_DIR}
