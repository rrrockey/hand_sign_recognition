import os
import json
import cv2

# paths
root = "../data"
images_root = os.path.join(root, "images")
labels_root = os.path.join(root, "labels")

# define label mapping
class_map = {
    "call": 0,
    "dislike": 1,
    "fist": 2,
    "four": 3,
    "like": 4,
    "mute": 5,
    "ok": 6,
    "one": 7,
    "palm": 8,
    "peace": 9,
    "peace_inverted": 10,
    "rock": 11,
    "stop": 12,
    "stop_inverted": 13,
    "three": 14,
    "three2": 15,
    "two_up": 16,
    "two_up_inverted": 17,
}

# class list
class_list = ["call",
             "dislike",
             "fist",
             "four",
             "like",
             "mute",
             "ok",
             "one",
             "palm",
             "peace",
             "peace_inverted",
             "rock",
             "stop",
             "stop_inverted",
             "three",
             "three2",
             "two_up",
             "two_up_inverted"]

split_list = ["train", "val", "test"]

for class_name in class_list:

    # iterate through train, val, test
    for split in split_list:

        # load one annotation file (repeat for train/val/test)
        ann_file = os.path.join(root, "labels", "ann_train_val", class_name+".json")
        with open(ann_file, "r") as f:
            annotations = json.load(f)


        for img_id, info in annotations.items():
            img_name = f"{img_id}.jpg"
            # print(img_name)
            img_path = os.path.join(images_root, split, class_name, img_name)
            print(img_name, " ", img_path, " ", labels_root, split)
            if not os.path.exists(img_path):
                # print(f"Skipping {img_name}, not found")
                continue

            # write label file
            label_dir = os.path.join(labels_root, split)
            os.makedirs(label_dir, exist_ok=True)
            label_path = os.path.join(label_dir, f"{img_id}.txt")

            with open(label_path, "w") as lf:
                for bbox, label in zip(info["bboxes"], info["labels"]):
                    if label not in class_map:
                        continue
                    class_id = class_map[label]
                    x_min, y_min, width, height = bbox  # values in COCO, need to be normalized to YOLO format
                    img = cv2.imread(img_path)
                    img_height, img_width, _ = img.shape

                    x_center = (x_min + width / 2) / img_width
                    y_center = (y_min + height / 2) / img_height
                    w_norm = width / img_width
                    h_norm = height / img_height


                    lf.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

print("✅ Done")



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