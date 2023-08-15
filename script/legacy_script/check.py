from glob import glob
import os

dir = "/home/ofotechjkr/workspace/tensorflow_OD/ofotech_train/models/03_feeder_pillar/dataset/temp_1/"

#subfolders = [ f.path for f in os.scandir(dir) if f.is_dir() ]

carry = glob(dir + "**/*.jpg", recursive = True)

print(carry)

print(len(carry))