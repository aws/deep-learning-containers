#!/bin/bash

echo "Cloning Tensorflow models directory (for conversion utilities)"
if [ ! -e tf-models ]; then
  git clone http://github.com/tensorflow/models tf-models
fi
(cd tf-models/research && protoc object_detection/protos/*.proto --python_out=.)

touch tf-models/__init__.py
touch tf-models/research/__init__.py

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

python3 prepare_coco_like_dataset.py

rm annotations_trainval2017.zip
rm -rf annotations

PYTHONPATH="tf-models:tf-models/research" python3 create_coco_tf_record.py   --logtostderr   --include_masks   --train_image_dir=new_train/   --val_image_dir=new_val/  --train_object_annotations_file=new_instances_train2017.json   --val_object_annotations_file=new_instances_val2017.json   --train_caption_annotations_file=new_captions_train2017.json   --val_caption_annotations_file=new_captions_val2017.json --output_dir=.

rm -rf new_train
rm -rf new_val

mkdir -p maskrcnn_tfrecords/annotations

mv train-00000-of-00001.tfrecord maskrcnn_tfrecords/train-00000-of-00256.tfrecord
mv val-00000-of-00001.tfrecord maskrcnn_tfrecords/val-00000-of-00032.tfrecord

mv new_captions_train2017.json maskrcnn_tfrecords/annotations/captions_train2017.json
mv new_captions_val2017.json maskrcnn_tfrecords/annotations/captions_val2017.json
mv new_instances_train2017.json maskrcnn_tfrecords/annotations/instances_train2017.json
mv new_instances_val2017.json maskrcnn_tfrecords/annotations/instances_val2017.json

cd maskrcnn_tfrecords
for i in {001..255}; do ln -s train-00000-of-00256.tfrecord train-00$i-of-00256.tfrecord; done
for i in {01..31}; do ln -s val-00000-of-00032.tfrecord val-000$i-of-00032.tfrecord; done
