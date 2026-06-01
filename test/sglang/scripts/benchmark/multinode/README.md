# Multi-Node SGLang Benchmark (DeepSeek V4 Pro)

Local testing procedure for running DeepSeek V4 Pro on 2x p5.48xlarge with EFA.
This is not yet automated in CI — use this guide to run manually.

## Prerequisites

- 2x p5.48xlarge instances in us-west-2b (same AZ, same subnet)
- 32 EFA interfaces per instance (all network cards configured as EFA/EFA-only)
- Security group with **all traffic self-referencing** on both ingress AND egress
- SGLang container image pulled on both nodes
- Model (`deepseek-v4-pro`, 742GB) extracted on both nodes at `/models/deepseek-v4-pro`

## Instance Launch

```bash
# Generate 32 EFA interface specs (card 0 = efa, cards 1-31 = efa-only)
python3 -c "
import json
interfaces = [{'NetworkCardIndex': 0, 'DeviceIndex': 0, 'Groups': ['<SG_ID>'], 'SubnetId': '<SUBNET_ID>', 'InterfaceType': 'efa'}]
for i in range(1, 32):
    interfaces.append({'NetworkCardIndex': i, 'DeviceIndex': 1, 'Groups': ['<SG_ID>'], 'SubnetId': '<SUBNET_ID>', 'InterfaceType': 'efa-only'})
with open('/tmp/p5-efa-interfaces.json', 'w') as f:
    json.dump(interfaces, f)
"

# Launch 2 instances from ODCR
aws ec2 run-instances \
  --image-id <DLAMI_AMI_ID> \
  --instance-type p5.48xlarge \
  --count 2 \
  --key-name <KEY_NAME> \
  --network-interfaces file:///tmp/p5-efa-interfaces.json \
  --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=<ODCR_ID>}" \
  --user-data file://userdata.sh \
  --region us-west-2
```

## Security Group Requirements

The security group MUST have:

- **Inbound**: All traffic (`-1`) from self (security group ID as source)
- **Outbound**: All traffic (`-1`) to self (security group ID as destination)

The self-referencing **egress** rule is critical for EFA — without it, NCCL gets
`Error: 15 (Unreachable remote)` even though instances are in the same subnet.

## NCCL Environment Variables

| Variable                   | Value     | Purpose                                                |
| -------------------------- | --------- | ------------------------------------------------------ |
| `NCCL_SOCKET_IFNAME`       | `enp71s0` | Bootstrap uses the IP-bearing interface (not EFA-only) |
| `FI_EFA_USE_DEVICE_RDMA`   | `1`       | Enable GPUDirect RDMA on p5                            |
| `NCCL_IB_DISABLE`          | `1`       | Disable InfiniBand (use EFA instead)                   |
| `FI_PROVIDER`              | `efa`     | Force EFA provider for libfabric                       |
| `SGLANG_SHARED_EXPERT_TP1` | `1`       | Fix V4 Pro partition size error at TP=16               |

## Start SGLang Server

Start worker (node 2) first, then leader (node 1):

```bash
IMAGE="<ECR_IMAGE>"
LEADER_IP="<NODE1_PRIVATE_IP>"

# Node 2 (worker)
docker run --gpus all --privileged --network=host --ipc=host \
  -e SGLANG_SHARED_EXPERT_TP1=1 \
  -e NCCL_SOCKET_IFNAME=enp71s0 \
  -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e NCCL_IB_DISABLE=1 \
  -e FI_PROVIDER=efa \
  -v /models/deepseek-v4-pro:/models \
  $IMAGE \
  --trust-remote-code \
  --model-path /models \
  --tp 16 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr ${LEADER_IP}:20000 \
  --moe-runner-backend marlin \
  --mem-fraction-static 0.9 \
  --host 0.0.0.0 \
  --port 30000

# Node 1 (leader) — same command but --node-rank 0
docker run --gpus all --privileged --network=host --ipc=host \
  -e SGLANG_SHARED_EXPERT_TP1=1 \
  -e NCCL_SOCKET_IFNAME=enp71s0 \
  -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e NCCL_IB_DISABLE=1 \
  -e FI_PROVIDER=efa \
  -v /models/deepseek-v4-pro:/models \
  $IMAGE \
  --trust-remote-code \
  --model-path /models \
  --tp 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr ${LEADER_IP}:20000 \
  --moe-runner-backend marlin \
  --mem-fraction-static 0.9 \
  --host 0.0.0.0 \
  --port 30000
```

Server takes ~12 minutes to start (model load + DeepGEMM JIT + CUDA graph capture).
Wait for `curl http://localhost:30000/health` to return 200 on the leader node.

## Run Benchmark

Run warmup first, then the real benchmark (from leader node):

```bash
# Warmup (triggers remaining JIT compilation)
docker run --rm --network=host --entrypoint python3 \
  -v /models/deepseek-v4-pro:/models \
  $IMAGE \
  -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 32 --model /models

sleep 10

# Real benchmark
docker run --rm --network=host --entrypoint python3 \
  -v /models/deepseek-v4-pro:/models \
  $IMAGE \
  -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --num-prompts 32 --model /models
```

## Expected Results (2x p5.48xlarge H100, TP=16, EFA)

| Metric                  | Value                                                  |
| ----------------------- | ------------------------------------------------------ |
| Output token throughput | ~43 tok/s                                              |
| Total token throughput  | ~230 tok/s                                             |
| Request throughput      | ~0.67 req/s                                            |
| Mean TPOT               | ~22.5 ms                                               |
| Mean TTFT               | ~23.6s (limited by sequential prefill due to KV cache) |
| Peak output throughput  | 92 tok/s                                               |

## Known Limitations

- **KV cache saturation**: With `--mem-fraction-static 0.9`, model weights use ~46.4GB/GPU
  leaving ~25.6GB for KV cache. The SWA cache saturates at 1 concurrent request, preventing batching.
- **TTFT is high**: Because only 1 request runs at a time, 32 requests queue up sequentially.
- **Placement group not required**: EFA works without a cluster placement group as long as
  instances are in the same AZ with correct security group rules.
- **NCCL_DEBUG=INFO slows startup**: Adds 20+ minutes to initialization. Only use for debugging.

## CI Integration (Future)

Options for automating this in GitHub Actions:

1. **EKS/LWS**: Deploy via kubectl from ubuntu-latest (public EKS endpoint). Requires RBAC + EKS access entry.
1. **Direct EC2**: Launch instances via AWS API, coordinate via SSM. Requires IAM role + FSx for model storage.
1. **EKS from in-cluster runner**: kubectl from gpu-l4-1gpu-runners (ARC). Only needs RBAC.
