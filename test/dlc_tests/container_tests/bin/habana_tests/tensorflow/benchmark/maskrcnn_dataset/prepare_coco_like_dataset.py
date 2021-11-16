import json
import subprocess
import time
import os
from multiprocessing import Pool, cpu_count
import requests
from PIL import Image


def check_response(url):
    start = time.time()
    output = True if requests.get(url, timeout=3).status_code == 200 else False
    print(time.time() - start)
    return output


def download_image(image):
    subprocess.run(["wget", image['url'], "-O", image['output_path'], "-P", image['mode']])


def download_images():
    with open("new_captions_val2017.json", "r") as cap_val, \
         open("new_captions_train2017.json", "r") as cap_train:
        captions_val = json.load(cap_val)
        captions_train = json.load(cap_train)
        urls_val = [{'url': i['flickr_url'], 'output_path': i['file_name'], 'mode': 'val'} for i in captions_val['images']]
        urls_train = [{'url': i['flickr_url'], 'output_path': i['file_name'], 'mode': 'train'} for i in captions_train['images']]

    with Pool(processes=6) as pool:
        print(pool.map(download_image, urls_val))

    with Pool(processes=6) as pool:
        print(pool.map(download_image, urls_train))


def prepare_jsons():
    with open("annotations/captions_val2017.json", "r") as cap_val, \
         open("annotations/captions_train2017.json", "r") as cap_train, \
         open("annotations/instances_val2017.json", "r") as ins_val, \
         open("annotations/instances_train2017.json", "r") as ins_train:
        captions_val = json.load(cap_val)
        captions_train = json.load(cap_train)
        instances_val = json.load(ins_val)
        instances_train = json.load(ins_train)

    captions_val['images'] = [i for i in captions_val['images'] if i['license'] == 4][:156]
    val_ids = set([i['id'] for i in captions_val['images']])
    captions_val['annotations'] = [a for a in captions_val['annotations'] if a['image_id'] in val_ids]

    captions_train['images'] = [i for i in captions_train['images'] if i['license'] == 4][:462]
    train_ids = set([i['id'] for i in captions_train['images']])
    captions_train['annotations'] = [a for a in captions_train['annotations'] if a['image_id'] in train_ids]

    instances_val['images'] = [i for i in instances_val['images'] if i['license'] == 4][:156]
    val_ids = set([i['id'] for i in instances_val['images']])
    instances_val['annotations'] = [a for a in instances_val['annotations'] if a['image_id'] in val_ids]

    instances_train['images'] = [i for i in instances_train['images'] if i['license'] == 4][:462]
    train_ids = set([i['id'] for i in instances_train['images']])
    instances_train['annotations'] = [a for a in instances_train['annotations'] if a['image_id'] in train_ids]

    with open("new_captions_train2017.json", "w") as cap_train, \
         open("new_captions_val2017.json", "w") as cap_val, \
         open("new_instances_val2017.json", "w") as ins_val, \
         open("new_instances_train2017.json", "w") as ins_train:
        json.dump(captions_val, cap_val)
        json.dump(captions_train, cap_train)
        json.dump(instances_val, ins_val)
        json.dump(instances_train, ins_train)


def create_images():
    with open("new_captions_val2017.json", "r") as cap_val, \
         open("new_captions_train2017.json", "r") as cap_train:
        captions_val = json.load(cap_val)
        captions_train = json.load(cap_train)
    subprocess.run(["wget", "https://freeforcommercialuse.net/wp-content/uploads/2021/03/msp_2102_7699.jpg", "-O", "img.jpg"])
    img = Image.open("img.jpg")
    for image in captions_val['images']:
        img_new = img.resize((image['width'], image['height']))
        img_new.save(f"new_val/{image['file_name']}")
    for image in captions_train['images']:
        img_new = img.resize((image['width'], image['height']))
        img_new.save(f"new_train/{image['file_name']}")


prepare_jsons()
os.makedirs("new_train")
os.makedirs("new_val")
create_images()

