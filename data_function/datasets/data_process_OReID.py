import os
import random
import shutil

pduke_train_occluded_dataset_path = os.path.join("/home/hy/project/data/Occluded_REID/train", "occluded_body_images/")
pduke_train_whole_dataset_path = os.path.join("/home/hy/project/data/Occluded_REID/train", "whole_body_images/")
pduke_test_occluded_dataset_path = os.path.join("/home/hy/project/data/Occluded_REID/test", "occluded_body_images/")
pduke_test_whole_dataset_path = os.path.join("/home/hy/project/data/Occluded_REID/test", "whole_body_images/")
new_dataset_path = "/home/hy/project/data/P-DukeMTMC-reid/format"

import os
import random
import shutil

occluded_dataset_path = os.path.join("/home/hy/project/data/Occluded_REID", "occluded_body_images/")
whole_dataset_path = os.path.join("/home/hy/project/data/Occluded_REID", "whole_body_images/")
new_dataset_path = "/home/hy/project/data/Occluded_REID/all"
os.makedirs(new_dataset_path)
for files in os.listdir(occluded_dataset_path):
    for img in os.listdir(os.path.join(occluded_dataset_path, files)):
        new_name = img.split(".")[0] + "_01" + ".tif"
        shutil.copy(os.path.join(occluded_dataset_path, files, img), os.path.join(new_dataset_path, new_name))

for files in os.listdir(whole_dataset_path):
    for img in os.listdir(os.path.join(whole_dataset_path, files)):
        new_name = img.split(".")[0] + "_02" + ".tif"
        shutil.copy(os.path.join(whole_dataset_path, files, img), os.path.join(new_dataset_path, new_name))

dataset_path = "/home/hy/project/data/Occluded_REID/all"
dataset_path_format = "/home/hy/project/data/Occluded_REID/format"
os.makedirs(dataset_path_format)
imgs = os.listdir(dataset_path)
for i in range(10):
    new_dataset_path = dataset_path_format + "/format_new" + str(i)
    pids = list(range(1, 201))
    train_pids = random.sample(pids, int(len(pids) / 2))
    os.makedirs(os.path.join(new_dataset_path, "bounding_box_train"))
    os.makedirs(os.path.join(new_dataset_path, "query"))
    os.makedirs(os.path.join(new_dataset_path, "bounding_box_test"))
    for img in imgs:
        pid, cid = img.split("_")[0], img.split("_")[2].split(".")[0]
        if int(pid) in train_pids:
            shutil.copy(os.path.join(dataset_path, img), os.path.join(new_dataset_path, "bounding_box_train", img))
        else:
            if int(cid) == 1:
                shutil.copy(os.path.join(dataset_path, img), os.path.join(new_dataset_path, "query", img))
            else:
                shutil.copy(os.path.join(dataset_path, img), os.path.join(new_dataset_path, "bounding_box_test", img))
