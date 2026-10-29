# Resource profile: gfnet-xs

| Metric | Value |
|---|---|
| Model | gfnet-xs |
| Parameters | 15.40 M (15,404,930) |
| FLOPs per slice (1 MAC = 1 FLOP, as in the paper) | 2.79 G |
| FFT FLOPs per slice (analytic estimate, not included above) | 0.037 G |
| Peak GPU VRAM, training | 2.23 GB (batch 64) |
| Peak GPU VRAM, inference | 0.47 GB (batch 64) |
| Inference latency, 1 slice | 6.23 ms |
| Inference latency, batch of 64 | 11.88 ms |
| Inference throughput | 5,388 slices/s |
| Inference time per patient (20 slices, batched) | 3.7 ms |
| Mean epoch time (train + validation) | 48.6 s |
| Total training time | 32.4 min (40 epochs) |
| Device | NVIDIA A100-PCIE-40GB (39 GB) |
| Mixed precision (FP16) | yes |
| PyTorch / CUDA | 2.13.0+cu130 / 13.0 |
