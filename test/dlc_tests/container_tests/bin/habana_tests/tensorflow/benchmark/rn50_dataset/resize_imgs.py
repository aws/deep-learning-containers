from PIL import Image
import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", required=True, type=str)
parser.add_argument("--out_dir", required=True, type=str)
args = parser.parse_args()

for num, file in enumerate(glob.glob(f"{args.in_dir}/*")):
    img = Image.open(file)
    size_w = np.random.randint(-300, 300)
    img = img.resize((470 + size_w, 400 + size_w))
    img.save(f"{args.out_dir}/image{num}.jpg")
