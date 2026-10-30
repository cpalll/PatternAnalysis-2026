# Resource profile: cnn

| Metric | Value |
|---|---|
| Model | cnn |
| Parameters | 1.17 M (1,173,154) |
| FLOPs per slice (1 MAC = 1 FLOP, as in the paper) | 2.56 G |
| FFT FLOPs per slice (analytic estimate, not included above) | 0.000 G |
| Peak GPU VRAM, training | 1.97 GB (batch 64) |
| Peak GPU VRAM, inference | 0.82 GB (batch 64) |
| Inference latency, 1 slice | 1.29 ms |
| Inference latency, batch of 64 | 8.84 ms |
| Inference throughput | 7,236 slices/s |
| Inference time per patient (20 slices, batched) | 2.8 ms |
| Mean epoch time (train + validation) | 45.5 s |
| Total training time | 30.3 min (40 epochs) |
| Device | NVIDIA A100-PCIE-40GB (39 GB) |
| Mixed precision (FP16) | yes |
| PyTorch / CUDA | 2.13.0+cu130 / 13.0 |
