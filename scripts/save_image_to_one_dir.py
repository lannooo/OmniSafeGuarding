from tqdm import tqdm
from module.load_datasets import *
from module.util import save_PIL_image


dataset = hf_load_MME()
for item in tqdm(dataset, "Saving MME..."):
    # print(item["image_path"])
    save_PIL_image(item["img"], item["image_path"])


# dataset = hf_load_genai_bench(split='midjourney')
# for item in tqdm(dataset, "Saving..."):
#     save_PIL_image(item["img"], item["image_path"])

# dataset = hf_load_mmbench(split='dev') + hf_load_mmbench(split='test')
# for item in tqdm(dataset, "Saving..."):
#     save_PIL_image(item["img"], item["image_path"])