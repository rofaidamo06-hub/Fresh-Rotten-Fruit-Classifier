import os
import shutil
import random

source = "Dataset"
destination = "Binary_Dataset"

classes = ["Fresh", "Rotten"]

IMAGES_PER_FRUIT = 300

for cls in classes:

    src_folder = os.path.join(source, cls)
    dst_folder = os.path.join(destination, cls)

    os.makedirs(dst_folder, exist_ok=True)

    for fruit in os.listdir(src_folder):

        fruit_path = os.path.join(src_folder, fruit)

        if os.path.isdir(fruit_path):

            images = os.listdir(fruit_path)

            random.shuffle(images)

            for img in images[:IMAGES_PER_FRUIT]:

                src_img = os.path.join(fruit_path, img)

                new_name = f"{fruit}_{img}"

                shutil.copy2(
                    src_img,
                    os.path.join(dst_folder, new_name)
                )

print("Done")
print("Fresh:", len(os.listdir("Binary_Dataset/Fresh")))
print("Rotten:", len(os.listdir("Binary_Dataset/Rotten")))
