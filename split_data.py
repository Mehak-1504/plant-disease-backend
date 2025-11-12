import os, shutil, random

# paths
src_images = "dataset/images"
src_masks = "dataset/masks"

val_ratio = 0.2  # 20% of data for validation

# create folders if missing
os.makedirs("dataset/val_images", exist_ok=True)
os.makedirs("dataset/val_masks", exist_ok=True)

# get all images
files = os.listdir(src_images)
random.shuffle(files)

val_count = int(len(files) * val_ratio)
val_files = files[:val_count]

# move validation files
for f in val_files:
    shutil.move(os.path.join(src_images, f), "dataset/val_images/" + f)
    shutil.move(os.path.join(src_masks, f), "dataset/val_masks/" + f)

print(f"✅ Moved {val_count} files to validation set.")
