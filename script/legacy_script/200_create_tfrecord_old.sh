mode=$1
# path='/home/ofotechjkr/workspace/selia_ai/road_hump'

echo "The current working directory: $PWD"
path="$(pwd)"

python /home/irad/workspace/irad_train/script/dataset_tools/create_tf_records.py \
        --data_dir=$path/dataset/$mode \
        --output_path=$path/dataset/$mode.tfrecord \
	--label_map_path=$path/dataset/labelmap.pbtxt
