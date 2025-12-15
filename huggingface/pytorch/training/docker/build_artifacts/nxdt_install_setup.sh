#!/usr/bin/env bash

set -o pipefail
set -e

# Install megatron-core
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.10.0
pip install .
cd megatron/core/datasets
make

VENV_PATH=$(pip show nemo_toolkit | grep Location | awk '{print $2}')
# Remove call to get_megatron_pretrained_bert_models() as it uses Transformer Engine which we don't support
sed -i 's/get_megatron_pretrained_bert_models()/[]/g' $VENV_PATH/nemo/collections/nlp/models/nlp_model.py
# Remove filepath checking as it could be an S3 path when S3 checkpointing
sed -i 's/ and self\._fs\.exists(ckpt_to_dir(filepath))//g' $VENV_PATH/nemo/utils/callbacks/nemo_model_checkpoint.py
