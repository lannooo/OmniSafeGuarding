from datasets import load_dataset
from module.load_datasets import hf_load_VLSafe

unsafe_set = hf_load_VLSafe()
# hf download coco ...
ds = load_dataset("path/to/coco")

images = [item['img'].split('/')[-1] for item in unsafe_set]
image_ids = set([int(item.split('.')[0]) for item in images])
print(len(image_ids))

selected_ds = ds['train'].filter(lambda x: x['image_id'] in image_ids)
print(len(selected_ds))

import os
from module.load_datasets import HF_VLSafe_Img_Dir

for item in selected_ds:
    # save image to local disk, e.g., ./data/LVLM_NLF/VLSafe/images/000000048848.jpg
    img_path = os.path.join(HF_VLSafe_Img_Dir, f"{item['image_id']:012d}.jpg")   # only jpg format
    # print(img_path)
    img_data = item["image"]  # PIL JpegImageFile save as jpg file
    img_data.save(img_path)
