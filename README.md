# TENSORFLOW OBJECT DETECTION API WORKFLOW
This comprehensive guide is designed to help you understand and implement object detection using TensorFlow's powerful Object Detection API. Whether you're a beginner or an experienced machine learning practitioner, this repository will walk you through the entire process, from preparing your dataset to training and inferring with your custom object detection model.

# GET STARTED

## 1. TRAINING PREPARATION (PREREQUISITE)
1. Assume that you already setup the Official Tensorflow OBJ in your conda environment.
If not, install the environment firsta from the following guide:
[Tensorflow 2 Object Detection API Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)

Enter the conda environment.
```
conda activate tfobj
```

2. Assume that you already setup and familiar with jupyter notebook. You shall install the jupyter extension in vscode.

## 1. DATASET PREPARATION
### 1.1 FOLDER PREP
1. inside /models, duplicate "model_name" as the main template. The folder shall contain:
	- dataset (folder)
	- debug (folder)
	- output (folder)
	- pseudo(folder)
	- training (folder)
	- 01_rename_space_with_underscore.sh
	- 02_move_unlabel.sh
	- 03_dataset_xml_to_csv.sh
	- 04_split_dataset_subdir.sh
	- 05_create_tfrecord.sh
	- 06_start_train.sh
	- 07_gen_model.sh
	- 08_start_eval.sh
	- 09_infer_draw_checkpoint.sh
	- 09_infer_draw_savedmodel.sh
	- 10_infer_pseudo.sh
	
2. rename model_template folder to "model name e.g bus_stop"

### 1.2 COLLECTION DATASET

* copy dataset into "dataset/images". The dataset can be in subdirectory.

### 1.3 CLEAN DATASET FOLDER AND FILE NAME

**1.3.1 Clean filename**
Need to ensure the folders and filenames do not have "space" and "symbols"
1. Open the ***01_rename_space_with_underscore.sh*** in text editor and change the variable accordingly
```
MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images 
```
2. run multiple times!! (some bugs, need to run multiple times):

```
sh 01_rename_space_with_underscore.sh 
```
**1.3.2 Remove Any Corrupted Images**
1. Open the ***02_check_corrupted_images.sh*** in text editor and change the variable accordingly
```
MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images
CORRUPT_IMAGE_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/corrupted_images
```
2. run

```
sh 02_check_corrupted_images.sh
```

**1.3.3 Remove Any XML File Error**
1. Open the ***02_check_xml_files_format.sh*** in text editor and change the variable accordingly
```
MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images
NO_LABEL_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/xml_error 
```
2. run

```
sh 02_check_xml_files_format.sh
```

**1.3.4 Remove Unlabeled Dataset**
1. Open the ***02_move_unlabel.sh*** in text editor and change the variable accordingly
```
MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images
NO_LABEL_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/no_label 
```
2. run

```
sh 02_move_unlabel.sh
```

### 1.4 DATASET EXPLORATION & ANALYSIS
We need to analyse and explore the labelled dataset.

**Generate dataset_labels.csv of the whole images dataset**
1. Open the ***03_dataset_xml_to_csv.sh*** in text editor and change the variable accordingly
```
MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images
CSV_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/ 
```
2. run:

```
sh 03_dataset_xml_to_csv.sh
```
You shall see dataset_labels.csv has been created in the dataset/ folder.

* cd into dataset/
* open the dataset_analytics.ipynb (You may open in jupyter notebook or directly in vscode with jupyter extension)

After executing the code in jupyter, you shall see details of the dataset. Please take note on:
a. The class count: This to ensure the dataset is balance. If the data not balance, we might need to balance the dataset first.
	* Option 1: Look for more data
	* Option 2: Upsampling data through data augmentation
b. Number of images: This to ensure how many numbers of images available for the training
c. Duplicates: Check if there is any duplication of data

3. Repeat the steps until you have a solid dataset.

## 2. MODEL TRAINING PREPARATION
### 2.1 DOWNLOAD THE TENSORFLOW PRETAINED MODEL
Choose and research accordlingly the suitable pretrained model for the user case and local server specification.
[TENSORFLOW 2 DETECTION MODEL ZOO](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

* Unzip the pretrained model. In general, it  shall contain the following files:
	1. checkpoint folder
	2. saved_model folder
	3. pipeline.config

* Copy the pretrained folder e.g "faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8" into model_name/training folder

### 2.2 SPLIT DATASET TO TRAIN AND TEST
* Split the dataset which located in dataset/images to training/train and training/test folder with default ratio 0.8.
* maintain the dataset strutcure from the datatse/images folder.

1. edit file ***04_split_dataset_subdir.sh***:
```
MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/dataset/images
TRAIN_DIR=$SERVER_DIR/models/$MODEL_DIR/training/train
TEST_DIR=$SERVER_DIR/models/$MODEL_DIR/training/test
RATIO=0.8
```
2. run:

```
sh 04_split_dataset_subdir.sh
```

Train and Test folder should have been created under model_name/dataset/training, containing 80% and 20% of the images (and *.xml files),respectively. 

### 2.3 DEFINED CLASSNAME
1. Edit the ***label_map.pbtxt*** files which located in _training/_ folder with the respective class id and name.

> [!IMPORTANT]
> Please ensure the class name is the same as in the .xml.

### 2.4 CREATE TFRECORD
DATASET_DIR is the location of the train and test folder.

1. edit file ***05_create_tfrecord.sh***:
```
MODEL_DIR=model_name
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
DATASET_DIR=$SERVER_DIR/models/$MODEL_DIR/training
```
2. run:

```
sh 05_create_tfrecord.sh
```

3. test.tf record & train.tf record will be created within the model_name/training folder. 
(check the file size, ensure the size is almost similar to the total size of train and test folder)

### 2.4 EDIT TRAINING CONFIG

1. Open the ***pipeline.config*** file which located in ***training/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/pipeline.config***

> [!IMPORTANT]
> num_classes (line 3) : change to total of object within the scope of the project
> fine_tune_checkpoint (line 100) : the path of ckpt-0 file
> train_input_reader/eval_input_reader(line 107 & 119) : edit labelmap and input_path
> batch_size = 8 (recommended)

## 3. START MODEL TRAINING

### 3.1 START CONDA ENV
* Open terminal in model_name directory
```
conda activate tfobj
```

### 3.2 MODEL TRAINING
1. Edit file ***06_start_train.sh***:

```	
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
MODEL="faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8"
NUM_TRAIN_STEPS=300000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
MODEL_DIR="output"
PIPELINE_CONFIG_PATH="training/$MODEL/pipeline.config"
```
> [!IMPORTANT]
> MODEL: must be the same name as the folder name of the downloaded pretrained model which already located in training/
> NUM_TRAIN_STEPS: set the suitable steps number

2. run:
```
sh 06_start_train.sh
```

### 3.3 MONITOR TRAINING WITH TENSORBOARD

1. run:
``` 
tensorboard --logdir=.
```
	copy link http://localhost:XXXX/ to browser

### 4. GENERATE MODEL
Upon completion of traning, we need to generate the model.

1. edit file ***07_gen_model.sh***:

```	
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
MODEL="faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8"
CHECKPOINT_DIR="output"
PIPELINE_CONFIG_PATH="training/$MODEL/pipeline.config"
OUTPUT_DIR="${CHECKPOINT_DIR}/model"
```

2. run:
```
sh 07_gen_model.sh 
```

* MODEL folder will be created inside OUTPUT folder
* OUTPUT : exported checkpoint and pb file to the specify directory
* generate checkpoint,saved_model and pipeline.config file

## 5. INFERENCE

### 5.1 INFERENCE WITH DRAW BBOX
1. Copy the dataset in debug/dataset

2. edit file ***09_infer_draw_checkpoint.sh***:
```
MODEL=model_name
THRESH=0.90
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
```
3. run:
```
	sh 09_infer_draw_checkpoint.sh
```


### 5.2 INFERENCE WITH XML GENERATED (PSEUDO LABEL)
This is useful to retrain model with additional dataset.

1. Copy the dataset in pseudo/dataset

> [!NOTE]
> Import to note that the THRESH is preferable low, so that we will train a new model with higher confidence level.

2. edit file ***10_infer_pseudo.sh***:
```
MODEL=model_name
THRESH=0.2
SERVER_DIR=/media/ofotechjkr/storage01/tf_object_detection
```
3. run:
```
	sh 10_infer_pseudo.sh
```

# FUTURE IMPROVMENT
1. Enable for eval visualization.
2. Enable for early stopping to avoid overfitting.
