#!/bin/bash

mkdir SQuAD
mkdir MRPC

echo "[INFO] Downloading SQuAD dataset"
cd SQuAD
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

echo "[INFO] Downloading MRPC dataset"
cd ../MRPC
wget https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt
wget https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt

cd ..
echo "[INFO] Processing MRPC dataset"
python3 process_bert_dataset.py --data_dir=. --task=MRPC --path_to_mrpc=./MRPC/

