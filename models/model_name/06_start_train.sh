SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
MODEL="faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8"
NUM_TRAIN_STEPS=300000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
MODEL_DIR="output"
PIPELINE_CONFIG_PATH="training/$MODEL/pipeline.config"

python $SERVER_DIR/script/object_detection/model_main_tf2.py \
  --model_dir=$MODEL_DIR \
  --num_train_steps=$NUM_TRAIN_STEPS \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
