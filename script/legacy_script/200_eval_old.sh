export CUDA_VISIBLE_DEVICES=-1
MODEL="centernet_hg104_512x512_coco17_tpu-8"
MODEL_DIR="output"
PIPELINE_CONFIG_PATH="dataset/$MODEL/pipeline.config"
CHECKPOINT_DIR=${MODEL_DIR}

python /home/irad/workspace/irad_train/script/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --eval_training_data=True \
    --alsologtostderr