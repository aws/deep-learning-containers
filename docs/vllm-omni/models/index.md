# Supported Models

All models listed below are regression-tested on every DLC vLLM-Omni release and work with the images listed on the [Overview](../index.md) page.

## Tested Models

| Modality | Model |
|---|---|
| **TTS** | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) |
| | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) (voice-clone) |
| | [FunAudioLLM/CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) (voice-clone) |
| **Image generation** | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| | [baidu/ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) |
| **Video generation** | [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) |
| | [Wan-AI/Wan2.1-VACE-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-Diffusers) |
| **Audio generation** | [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) |
| **Omni chat** | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) |

## Model Compatibility

Any model supported by upstream vLLM-Omni should work. Requirements:

- Models must have a standard HuggingFace `config.json` with a recognized `model_type`, or be diffusers pipeline models with `model_index.json`.
- Multi-stage omni models (thinker + talker + decoder) like Qwen2.5-Omni need significantly more VRAM than the model size suggests. Refer to individual model cards for minimum GPU requirements.
