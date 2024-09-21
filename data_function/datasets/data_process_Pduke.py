import os
import random
import shutil

pduke_train_occluded_dataset_path = os.path.join("/home/hy/project/data/P-DukeMTMC-reid/train", "occluded_body_images/")
pduke_train_whole_dataset_path = os.path.join("/home/hy/project/data/P-DukeMTMC-reid/train", "whole_body_images/")
pduke_test_occluded_dataset_path = os.path.join("/home/hy/project/data/P-DukeMTMC-reid/test", "occluded_body_images/")
pduke_test_whole_dataset_path = os.path.join("/home/hy/project/data/P-DukeMTMC-reid/test", "whole_body_images/")
new_dataset_path = "/home/hy/project/data/P-DukeMTMC-reid/format"
os.makedirs(os.path.join(new_dataset_path, "bounding_box_train"))
os.makedirs(os.path.join(new_dataset_path, "query"))
os.makedirs(os.path.join(new_dataset_path, "bounding_box_test"))
for files in os.listdir(pduke_train_occluded_dataset_path):
    for img in os.listdir(os.path.join(pduke_train_occluded_dataset_path, files)):
        new_name = img.split(".")[0] + "_01" + ".jpg"
        shutil.copy(os.path.join(pduke_train_occluded_dataset_path, files, img), os.path.join(new_dataset_path, "bounding_box_train", new_name))

for files in os.listdir(pduke_train_whole_dataset_path):
    for img in os.listdir(os.path.join(pduke_train_whole_dataset_path, files)):
        new_name = img.split(".")[0] + "_02" + ".jpg"
        shutil.copy(os.path.join(pduke_train_whole_dataset_path, files, img), os.path.join(new_dataset_path, "bounding_box_train", new_name))

for files in os.listdir(pduke_test_occluded_dataset_path):
    for img in os.listdir(os.path.join(pduke_test_occluded_dataset_path, files)):
        new_name = img.split(".")[0] + "_01" + ".jpg"
        shutil.copy(os.path.join(pduke_test_occluded_dataset_path, files, img), os.path.join(new_dataset_path, "query", new_name))

for files in os.listdir(pduke_test_whole_dataset_path):
    for img in os.listdir(os.path.join(pduke_test_whole_dataset_path, files)):
        new_name = img.split(".")[0] + "_02" + ".jpg"
        shutil.copy(os.path.join(pduke_test_whole_dataset_path, files, img), os.path.join(new_dataset_path, "bounding_box_test", new_name))
