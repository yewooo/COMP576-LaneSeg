import os
import random
import shutil

SOURCE = "F:/COMP576/final/archive/yolo_data"
TARGET = "F:/COMP576/final/culane_6k"

TRAIN_NUM = 6000
VAL_NUM = 1000

for split in ["train", "val"]:
    os.makedirs(f"{TARGET}/images/{split}", exist_ok=True)
    os.makedirs(f"{TARGET}/labels/{split}", exist_ok=True)

train_images = os.listdir(f"{SOURCE}/train/images")
test_images = os.listdir(f"{SOURCE}/test/images")

train_sample = random.sample(train_images, TRAIN_NUM)
val_sample = random.sample(test_images, VAL_NUM)

def copy_subset(sample_list, source_split, target_split):
    for img in sample_list:
        src_img = f"{SOURCE}/{source_split}/images/{img}"
        dst_img = f"{TARGET}/images/{target_split}/{img}"
        shutil.copy(src_img, dst_img)

        label = img.replace(".jpg", ".txt")
        src_label = f"{SOURCE}/{source_split}/labels/{label}"
        dst_label = f"{TARGET}/labels/{target_split}/{label}"
        shutil.copy(src_label, dst_label)

copy_subset(train_sample, "train", "train")
copy_subset(val_sample, "test", "val")

print("CULane 6k subset created successfully!")
