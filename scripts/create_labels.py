import os
import json
import cv2

# paths
img_root = "../data/images"
label_root = "../data/labels"
anno_dir = "../data/labels/ann_train_val"

# map your gesture labels to IDs
class_map = {
    "call": 0,
    "dislike": 1,
    "fist": 2,
    # ...
}

for ann_file in os.listdir(anno_dir):
    if not ann_file.endswith(".json"):
        continue
    
    with open(os.path.join(anno_dir, ann_file), "r") as f:
        annotations = json.load(f)
    
    for ann in annotations:  # depends on JSON structure
        img_path = os.path.join(img_root, ann["split"], ann["file_name"])
        
        if not os.path.exists(img_path):
            print(f"Skipping missing image: {img_path}")
            continue
        
        # read image size
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # convert bbox
        x_min, y_min, box_w, box_h = ann["bbox"]
        x_center = (x_min + box_w/2) / w
        y_center = (y_min + box_h/2) / h
        norm_w   = box_w / w
        norm_h   = box_h / h

        class_id = class_map[ann["category"]]  # adjust depending on field name

        # save label
        label_dir = os.path.join(label_root, ann["split"])
        os.makedirs(label_dir, exist_ok=True)
        label_path = os.path.join(label_dir, os.path.splitext(ann["file_name"])[0] + ".txt")

        with open(label_path, "a") as out:
            out.write(f"{class_id} {x_center} {y_center} {norm_w} {norm_h}\n")



# import os, shutil, json

# SRC_ROOT = '../data/labels/ann_train_val/'
# DEST_ROOT = '../data/'

# splits = ['test', 'train', 'val']

# def read_json():
#     with open(SRC_ROOT+'call.json', 'r') as f:
#         data = json.load(f)
        
#         for uuid, info in data.items():
#             print(f"ID: {uuid}")
#             print(f"  Labels: {info['labels']}")
#             print(f"  BBoxes: {info['bboxes']}")
#             print(f"  Leading hand: {info['leading_hand']} (conf {info['leading_conf']})")
#             print(f"  User: {info['user_id']}")
#             print()

    


# if __name__ == '__main__':
#     read_json()
