import os
import shutil
import random
from pycocotools.coco import COCO

coco_images_path = "D:/Downloads/data/coco_images/train2017"
output_path = "D:/Downloads/data/coco_non_lesion"
coco_annotation_file = "D:/Downloads/data/annotations/instances_train2017.json"

images_per_category = 500
categories_to_use = [
    "dog", "cat", "car", "bottle", "chair", "person", "bicycle",
    "airplane", "book", "bird", "laptop", "train", "tv", "bus"
]

os.makedirs(output_path, exist_ok=True)

# Use the COCO annotations JSON
coco = COCO(coco_annotation_file)

if categories_to_use is None:
    all_cats = coco.loadCats(coco.getCatIds())
    categories_to_use = [c['name'] for c in random.sample(all_cats, 15)]

print("Using categories:", categories_to_use)

# Get all category IDs
cat_ids = coco.getCatIds()
for cat_name in categories_to_use:
    cat_ids = coco.getCatIds(catNms=[cat_name])
    img_ids = coco.getImgIds(catIds=cat_ids)
    random.shuffle(img_ids)
    selected = img_ids[:images_per_category]

    for img_id in selected:
        img_info = coco.loadImgs(img_id)[0]
        src_path = os.path.join(coco_images_path, img_info['file_name'])
        dst_path = os.path.join(output_path, img_info['file_name'])
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)