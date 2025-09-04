import os, random, shutil

random.seed(42)  # reproducibility

SRC_ROOT = "../data/hagrid-sample-120k-384p/hagrid_120k"   # directory where your train_val_* folders live
DEST_ROOT = "../data/images"  # output base directory

splits = {"train": 0.7, "val": 0.15, "test": 0.15}

def split_class(class_dir):
    files = os.listdir(class_dir)
    random.shuffle(files)

    n = len(files)
    train_end = int(splits["train"] * n)
    val_end = train_end + int(splits["val"] * n)

    split_files = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    # class name is "call", "fist", etc.
    class_name = os.path.basename(class_dir).replace("train_val_", "")
    print(f"Processing {class_name}: {n} files")

    for split, f_list in split_files.items():
        outdir = os.path.join(DEST_ROOT, split, class_name)
        os.makedirs(outdir, exist_ok=True)
        for f in f_list:
            shutil.move(os.path.join(class_dir, f), os.path.join(outdir, f))


if __name__ == "__main__":
    for dirname in os.listdir(SRC_ROOT):
        if dirname.startswith("train_val_"):
            split_class(os.path.join(SRC_ROOT, dirname))

    print("✅ Finished splitting all classes into train/val/test.")

