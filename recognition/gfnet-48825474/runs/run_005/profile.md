# Resource profile: gfnet-h-ti

| Metric | Value |
|---|---|
| Model | gfnet-h-ti |
| Parameters | 14.57 M (14,571,074) |
| FLOPs per slice (1 MAC = 1 FLOP, as in the paper) | 2.03 G |
| FFT FLOPs per slice (analytic estimate, not included above) | 0.075 G |
| Peak GPU VRAM, training | 3.58 GB (batch 64) |
| Peak GPU VRAM, inference | 0.64 GB (batch 64) |
| Inference latency, 1 slice | 10.63 ms |
| Inference latency, batch of 64 | 20.87 ms |
| Inference throughput | 3,066 slices/s |
| Inference time per patient (20 slices, batched) | 6.5 ms |
| Mean epoch time (train + validation) | 52.5 s |
| Total training time | 35.0 min (40 epochs) |
| Device | NVIDIA A100-PCIE-40GB (39 GB) |
| Mixed precision (FP16) | yes |
| PyTorch / CUDA | 2.13.0+cu130 / 13.0 |
