# Embedding Serving using TEI DLC

Production-ready Docker images for serving embedding, reranker, and sequence-classification models with
[Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) on {{ sagemaker }}. Built and published by
[Hugging Face](https://huggingface.co) in collaboration with {{ aws }}.

TEI is a high-performance toolkit written in Rust for serving text embedding models, with dynamic batching, optimized transformer kernels (using Flash
Attention, Candle, and cuBLASLt), and small, fast-booting images.

## Images

| Accelerator | Image (`us-east-1` example) | Default Port |
| --- | --- | --- |
| GPU | `683313688378.dkr.ecr.us-east-1.amazonaws.com/tei:2.0.1-tei1.8.2-gpu-py310-cu122-ubuntu22.04` | 8080 |
| CPU | `683313688378.dkr.ecr.us-east-1.amazonaws.com/tei-cpu:2.0.1-tei1.8.2-cpu-py310-ubuntu22.04` | 8080 |

Unlike most {{ dlc_long }}, the TEI images are hosted in a different {{ ecr_short }} account in each region. The simplest way to get the right URI is
the SageMaker Python SDK helper, which resolves the account for your session's region:

```python
from sagemaker.huggingface import get_huggingface_llm_image_uri

gpu_image = get_huggingface_llm_image_uri("huggingface-tei", version="1.8.2")
cpu_image = get_huggingface_llm_image_uri("huggingface-tei-cpu", version="1.8.2")
```

The per-region registry account IDs are listed below.

| Region | Account ID |
| --- | --- |
| `af-south-1` | 510948584623 |
| `ap-east-1` | 651117190479 |
| `ap-northeast-1` | 354813040037 |
| `ap-northeast-2` | 366743142698 |
| `ap-northeast-3` | 867004704886 |
| `ap-south-1` | 720646828776 |
| `ap-south-2` | 628508329040 |
| `ap-southeast-1` | 121021644041 |
| `ap-southeast-2` | 783357654285 |
| `ap-southeast-3` | 951798379941 |
| `ap-southeast-4` | 106583098589 |
| `ca-central-1` | 341280168497 |
| `ca-west-1` | 190319476487 |
| `cn-north-1` | 450853457545 |
| `cn-northwest-1` | 451049120500 |
| `eu-central-1` | 492215442770 |
| `eu-central-2` | 680994064768 |
| `eu-north-1` | 662702820516 |
| `eu-south-1` | 978288397137 |
| `eu-south-2` | 104374241257 |
| `eu-west-1` | 141502667606 |
| `eu-west-2` | 764974769150 |
| `eu-west-3` | 659782779980 |
| `il-central-1` | 898809789911 |
| `me-central-1` | 272398656194 |
| `me-south-1` | 801668240914 |
| `sa-east-1` | 737474898029 |
| `us-east-1` | 683313688378 |
| `us-east-2` | 257758044811 |
| `us-gov-east-1` | 237065988967 |
| `us-gov-west-1` | 414596584902 |
| `us-iso-east-1` | 833128469047 |
| `us-isob-east-1` | 281123927165 |
| `us-west-1` | 746614075791 |
| `us-west-2` | 246618743249 |

## API Endpoints

On {{ sagemaker }}, all traffic goes through `POST /invocations` (with `GET /ping` for health checks). At startup, the container binds `/invocations`
to the task matching the loaded model type:

| Model Type | Task | Payload |
| --- | --- | --- |
| Embedding | Embeddings | `{"inputs": "..."}` or `{"inputs": ["...", "..."]}` |
| Reranker | Reranking | `{"query": "...", "texts": ["...", "..."]}` |
| Classifier | Sequence classification | `{"inputs": "..."}` |

Refer to the [TEI API documentation](https://huggingface.github.io/text-embeddings-inference/) for request/response schemas.

## How They're Built

- **Released with TEI** — image versions track upstream [TEI releases](https://github.com/huggingface/text-embeddings-inference/releases) and are
  published by Hugging Face to the regional SageMaker registries.
- **Discoverable via the SageMaker SDK** — current versions are registered in the
  [SageMaker Python SDK image URI config](https://github.com/aws/sagemaker-python-sdk/blob/master-v2/src/sagemaker/image_uri_config/huggingface-tei.json),
  so `get_huggingface_llm_image_uri` always resolves a valid URI.

For deployment walkthroughs, see [{{ sagemaker }} Deployment](deployment/sagemaker.md).
