#!/bin/bash

RAW_DIR="raw_images"
PROCESSED_DIR="processed_images"
TFRECORD_DIR="tf_records"

if test -d "$RAW_DIR/background"; then
    echo "[INFO] $RAW_DIR exists, please choose another directory to write images to"
    exit 0
fi
if test -d "$PROCESSED_DIR/background"; then
    echo "[INFO] $PROCESSED_DIR exists, please choose another directory to write images to"
    exit 0
fi
if test -d "$TFRECORD_DIR"; then
    echo "[INFO] `pwd`/$TFRECORD_DIR exists, please choose another directory to write images to"
    exit 0
fi

mkdir -p $RAW_DIR/background
echo "[INFO] Created `pwd`/$RAW_DIR/background"

cd $RAW_DIR/background
echo "[INFO] Start downloading images"
for i in {1..11}; do wget -q -nd -H -p -A jpg,jpeg,png -R "logo*" -e robots=off https://freeforcommercialuse.net/page/$i; done
echo "[INFO] Finished downloading images"
cd -

mkdir -p $PROCESSED_DIR/background
echo "[INFO] Created `pwd`/$PROCESSED_DIR/background"

echo "[INFO] Resizing images"
python3 resize_imgs.py --in_dir=$RAW_DIR/background --out_dir=$PROCESSED_DIR/background
echo "[INFO] Resizing images completed"

cp -r $PROCESSED_DIR/background $PROCESSED_DIR/kite
cp -r $PROCESSED_DIR/background $PROCESSED_DIR/vulture

echo `pwd`

echo "[INFO] Creating tfrecords"
echo "Running command: python3 tfrecords.py --output_directory=.  --train_directory=$PROCESSED_DIR --lables_file=./labels.txt"
python3 tfrecords.py --output_directory=. --train_directory=$PROCESSED_DIR --labels_file=./labels.txt

mkdir -p $TFRECORD_DIR/img_train
mkdir -p $TFRECORD_DIR/img_val

echo "[INFO] Created `pwd`/$TFRECORD_DIR/img_train"
echo "[INFO] Created `pwd`/$TFRECORD_DIR/img_val"

cp train-00000-of-00001 $TFRECORD_DIR/img_train/img_train-00000-of-01024
cp train-00000-of-00001 $TFRECORD_DIR/img_val/img_val-00000-of-00128

echo "[INFO] Creating symlinks"

cd $TFRECORD_DIR/img_train
for i in {0001..1023}; do ln -s img_train-00000-of-01024 img_train-0$i-of-01024; done
cd ../img_val
for i in {001..128}; do ln -s img_val-00000-of-00128 img_val-00$i-of-00128; done

echo "[INFO] Created symlinks"

echo "In img_valid there are `ls | wc -l` files"
cd ../img_train
echo "In img_train there are `ls | wc -l` files"

echo "[INFO] Creating dataset finished."
