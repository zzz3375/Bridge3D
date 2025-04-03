import os
import glob
import random
import shutil

# Set paths
dataset_path = r"DJI_202411231419_071"  # Adjust to your dataset location
label_path = "labels"
output_path = r"C:\Users\13694\yolov8-SEA\datasets\Bridge-componient"

# Class names
class_names = [
    "Nonbridge", "Slab", "Beam", "Column", "Nonestructural",
    "Rail", "Sleeper", "Concrete damage", "Exposed rebar", "Nondamage"
]

# Create directories
for split in ["train", "test"]:
    os.makedirs(os.path.join(output_path, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", split), exist_ok=True)

# Get all images
images = glob.glob(os.path.join(dataset_path, "*.JPG"))  # Adjust for PNGs if needed

# Shuffle and split
random.seed(42)  # For reproducibility
random.shuffle(images)

split_ratio = 0.8
train_size = int(len(images) * split_ratio)

train_images = images[:train_size]
test_images = images[train_size:]

# Move files
def move_files(image_list, split):
    for img in image_list:
        label = os.path.join(label_path, os.path.basename(img).replace(".JPG", ".txt"))
        shutil.copy(img, os.path.join(output_path, "images", split, os.path.basename(img)))
        if os.path.exists(label):
            shutil.copy(label, os.path.join(output_path, "labels", split, os.path.basename(label)))

move_files(train_images, "train")
move_files(test_images, "test")

# Generate data.yaml
yaml_content = f"""train: {os.path.join(output_path, 'images', 'train')}
val: {os.path.join(output_path, 'images', 'test')}

nc: {len(class_names)}
names: {class_names}
"""

yaml_path = os.path.join(output_path, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"Dataset split completed! data.yaml saved at {yaml_path}")
