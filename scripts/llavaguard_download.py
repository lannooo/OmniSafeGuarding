import os
import hashlib
import shutil
import requests
import pandas as pd

from urllib.parse import urlparse
from PIL import Image
from datasets import load_dataset

def generate_uid_from_url(url, length=32):
    hash_object = hashlib.sha256()
    hash_object.update(url.encode('utf-8'))
    uid = hash_object.hexdigest()
    return uid[:length]

def parse_filename_from_url(url):
    res = urlparse(url.strip())
    img_name = res.path.split("/")[-1]
    return img_name


save_dir = 'tmp'
cache_dir = 'download_cache'
url_uid_file = './url_uid.csv'
if not os.path.exists(url_uid_file):
    pd.DataFrame(columns=['url', 'uid']).to_csv(url_uid_file, index=False)

url_uid_df = pd.read_csv(url_uid_file)

# load all urls
urls = []
dataset = load_dataset("data/LlavaGuard")
for subset in ['train', 'test']:
    for url in dataset[subset]['url']:
        urls.append(url)
# remove redundant urls
urls = list(set(urls))

error_urls = []
existing_uids = set(url_uid_df['uid'].tolist())


# requests headers to get images
headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
}

for i, url in enumerate(urls):
    uid = generate_uid_from_url(url)

    # skip if uid already exists in url_uid_df
    if uid in existing_uids:
        print(f"{uid}: {url} already processed, skip")
        continue

    if "huggingface.co" in url:
        # if url from huggingface.co, e.g., https://huggingface.co/datasets/AIML-TUDA/smid/resolve/main/images/b13_p233_19.jpg
        # fetch from local directory, e.g., data/smid/images/b13_p233_19.jpg
        if "AIML-TUDA/smid" in url:
            parent_dir = "data/smid/images"
        else:
            print("Unknown URL from huggingface.co: " + url)
            error_urls.append((url, uid))
            continue
        # parse image filename from url
        filename = url.split("/")[-1]
        ext = filename.split(".")[-1]
        save_file = f"{uid}.{ext}"

        local_file_path = os.path.join(parent_dir, filename)
        save_file_path = os.path.join(save_dir, save_file)

        # copy image file from local_file path to save_file path
        shutil.copyfile(local_file_path, save_file_path)
        print(f"Copied {local_file_path} to {save_file_path}")

        # add a new row to url_uid_df
        url_uid_df.loc[len(url_uid_df)] = [url, uid]

    else:
        # if url from external sources, e.g., https://libertywingspan.com/wp-content/uploads/2016/02/marydude.jpg
        # download image file with PIL.Image, and then save to local directory
        try:
            img = Image.open(requests.get(url, stream=True, timeout=5, headers=headers, allow_redirects=True).raw)
        except Exception as e:
            # Errors can be complex, for example: 
            # 1) the image cannot be found (404)
            # 2) the website is checking live person
            # 3) the website has to be logged in (such as Reddit)
            # 4) Timeout
            # First, we try to load from cached files (which are downloaded previously)
            # find by name
            file_name = parse_filename_from_url(url)
            local_cache_file = os.path.join(cache_dir, file_name)
            if os.path.exists(local_cache_file):
                img = Image.open(local_cache_file)
            else:
                print("Error when opening image: " + url)
                error_urls.append((url, uid))
                continue

        if img.format == 'JPEG':
            ext = 'jpg'
        elif img.format == 'PNG':
            ext = 'png'
        elif img.format == 'WEBP':
            ext = 'webp'
        else:
            print("Unknown image format: " + img.format)
            error_urls.append((url, uid))
            continue

        save_file = f"{uid}.{ext}"
        save_file_path = os.path.join(save_dir, save_file)

        # save Image to local image file
        img.save(save_file_path)
        print(f"Saved {save_file_path}")
        # append a new row to url_uid_df
        url_uid_df.loc[len(url_uid_df)] = [url, uid]

# update url_uid file
url_uid_df.to_csv(url_uid_file, index=False)
print("Updated url_uid.csv file")

# print error urls:
print("Error urls:")
for url, uid in error_urls:
    print(uid, ":", url)