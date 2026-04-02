#!/bin/bash
# Smoke test for vLLM-Omni EC2 images
# Validates that omni models can load and produce output
set -eux

nvidia-smi

MODEL_PATH="${1:?Usage: $0 <model-path> <model-type>}"
MODEL_TYPE="${2:?Usage: $0 <model-path> <model-type>}"

echo "=== Testing vLLM-Omni: ${MODEL_TYPE} model at ${MODEL_PATH} ==="

if [ "${MODEL_TYPE}" = "tts" ]; then
    # Qwen3-TTS offline inference test
    python3 -c "
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model='${MODEL_PATH}', stage_init_timeout=600)
additional_information = {
    'task_type': ['CustomVoice'],
    'text': ['Hello, this is a test of the text to speech system.'],
    'language': ['English'],
    'speaker': ['Ryan'],
    'instruct': [''],
    'max_new_tokens': [2048],
}
inputs = {
    'prompt_token_ids': [0] * 512,
    'additional_information': additional_information,
}
outputs = omni.generate([inputs])
for out in outputs:
    mm = out.request_output.outputs[0].multimodal_output
    assert 'audio' in mm, 'No audio in output'
    assert mm['sr'], 'No sample rate in output'
    print(f'Audio generated: sr={mm[\"sr\"]}, chunks={len(mm[\"audio\"])}')
print('TTS smoke test PASSED')
"

elif [ "${MODEL_TYPE}" = "diffusion" ]; then
    # FLUX.2-klein image generation test
    python3 -c "
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model='${MODEL_PATH}', stage_init_timeout=600)
prompt = 'a red apple on a white table'
outputs = omni.generate(prompt)
images = outputs[0].request_output.images
assert len(images) > 0, 'No images generated'
images[0].save('/tmp/omni_test_output.png')
assert os.path.exists('/tmp/omni_test_output.png'), 'Output image not saved'
size = os.path.getsize('/tmp/omni_test_output.png')
assert size > 1000, f'Output image too small: {size} bytes'
print(f'Image generated: {images[0].size}, file size: {size} bytes')
print('Diffusion smoke test PASSED')
"

else
    echo "ERROR: Unknown model type: ${MODEL_TYPE}"
    exit 1
fi

echo "=== vLLM-Omni ${MODEL_TYPE} test PASSED ==="
