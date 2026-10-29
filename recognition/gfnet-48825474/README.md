# GFNet for classification of Alzheimer's on the ADNI dataset

This repository implements the GFNet model by Rao et al. and assesses its effectiveness for classifying Alzheimer's disease based on brain MRI scans from the Alzheimer's Disease Neuroimagin Initiative (ADNI) dataset. It covers the full data preparation, training, validation, testing, and evaluation pipeline for the GFNet model. 

## About the GFNet model

Global Filter Networks (GFNet), introduced by Rao et al. (NeurIPS 2021), are a vision architecture that replaces the self-attention layer of vision transformers with a global filter layer. An input image is divided into non-overlapping 16 x 16 patches, each linearly projected into a token. In every GFNet block, the grid of tokens is transformed into the frequency domain with a 2D fast Fourier transform (FFT), multiplied element-wise by learnable global filters, and transformed back with an inverse FFT. This is equivalent to a depthwise convolution whose kernel spans the whole image, so every token interacts with every other. The cost, however, grows only log-linearly, $O(N \log N)$, rather than quadratically as in attention. A feed-forward network then mixes information across feature channels. On ImageNet, GFNet achieves accuracy competitive with transformer and MLP-style models of similar size, with favourable efficiency, generalisation and robustness.

## GFNet Architecture

### GFNet Advantages

### GFNet Disadvantages

---

## Quick Start

### Dependencies

| Package | Version used | Used for |
|---|---|---|
| Python | 3.11 | |
| PyTorch | 2.13.0 (CUDA 13.0 build, `+cu130`) | Model, training, inference |
| torchvision | TBC | Data transforms (`transforms.v2`) |
| Pillow | TBC | Loading JPEG slices |
| NumPy | TBC | Data loading and plotting only (not used by the model) |
| matplotlib | TBC | Plots |

```bash
conda create -n gfnet python=3.11
conda activate gfnet
pip install torch==2.13.0 torchvision==<TBC> --index-url https://download.pytorch.org/whl/cu130
pip install pillow==<TBC> numpy==<TBC> matplotlib==<TBC>
```

### Reproducibility

- **Fixed seed.** `--seed` (default 42) seeds Python and PyTorch, and also fixes the patient-level train/validation split, so the same command always uses the same patients in each split.
- **Fixed held-out test set.** The provided `test/` folder is never used for training, model selection, calibration or choosing the referral threshold; it is evaluated once at the end of each run.
- **Everything needed to reproduce a run is saved with it.** Each `runs/run_NNN/` stores the full command-line arguments (`test_metrics.json`), the normalisation statistics and fitted temperature (inside `best.pt`), per-epoch history and a `profile.md` recording the GPU, PyTorch and CUDA versions.
- **Not bit-for-bit deterministic.** GPU kernels (cuDNN, cuFFT) and mixed-precision arithmetic are not fully deterministic, so re-running the same command can differ slightly in the last decimal places of the metrics. Differences between runs smaller than about one patient (0.002 patient-level accuracy) should be treated as noise.
- **Reproducing the reported run** (`run_004`, GFNet-XS, about 32 minutes on an A100):

```bash
python train.py --model gfnet-xs --drop-path 0.3 --epochs 40 --batch-size 64 --seed 42 --cache
python predict.py --checkpoint runs/run_NNN/best.pt
```

### Usage

Run all commands from `recognition/gfnet-48825474/`. `--root` points to the `AD_NC` folder (default: `/home/groups/comp3710/ADNI/AD_NC` on Rangpur).

#### Train, validate and test: `train.py`

```bash
python train.py --root ../../AD_NC --model gfnet-xs --epochs 60 --batch-size 64 --cache
```

One run trains the model, validates every epoch, keeps the best checkpoint, fits a softmax temperature on the validation set, then evaluates once on the held-out test set.

| Option | Default | Purpose |
|---|---|---|
| `--model` | `gfnet-xs` | `gfnet-ti`, `gfnet-xs`, `gfnet-s`, `gfnet-b`, `gfnet-h-ti`, `gfnet-h-s`, `gfnet-h-b` |
| `--epochs` / `--batch-size` | `60` / `64` | Lower the batch size if GPU memory runs out (the LR scales with it) |
| `--lr` | `1e-3 x batch/512` | Peak learning rate (warmup then cosine decay) |
| `--pretrained` | none | Official GFNet ImageNet checkpoint to fine-tune from |
| `--cache` | off | Decode all slices into RAM once (~2 GB) for faster epochs |
| `--patience` | `15` | Early-stopping patience in epochs |
| `--out-dir` | `runs/gfnet` | Where outputs are written |
| `--limit-batches` | `0` | Debug: cap batches per epoch, e.g. `5` for a quick smoke test |

Outputs in `--out-dir`:

- `best.pt`, `last.pt`: checkpoints (weights, model config, normalisation stats, temperature)
- `training_curves.png`: loss, accuracy, AUROC/macro-F1 and LR, redrawn every epoch
- `history.csv`: per-epoch metrics
- `test_results.png`: ROC, reliability diagram, risk-coverage curve, confusion matrix
- `test_metrics.json`: final test metrics, calibration, selective-classification and compute cost
- `test_predictions.csv`: per-patient test probabilities and correctness

#### Predict: `predict.py`

```bash
python predict.py --checkpoint runs/gfnet/best.pt                        # held-out test set
python predict.py --checkpoint runs/gfnet/best.pt --target-acc 0.9       # looser referral rule
python predict.py --checkpoint runs/gfnet/best.pt --images path/to/slices/   # unlabelled slices
```

Loads the trained model, picks a referral threshold on the validation patients (the lowest confidence at which accuracy on auto-decided patients meets `--target-acc`, default 0.95), then classifies each test patient as AD, NC or REFER (routed to a human examiner). It prints accuracy, coverage, how many errors the referral catches and the least confident patients.

Outputs in `<checkpoint dir>/predict/`:

- `examples.png`: confident, misclassified and referred patients, each with an evidence heatmap (red = evidence for AD, blue = evidence for NC)
- `confidence.png`: confidence of correct vs incorrect patients with the referral threshold
- `global_filters.png`: learned global filter magnitudes per block
- `predictions.csv`: per-patient P(AD), confidence and decision

## Results

Example inputs, outputs, and visual plots of the GFNet model.

### Example inputs, outputs and plots

#### Input

Each patient is 20 greyscale 2D sagittal MRI slices (240 x 256 JPEG, named `{patientID}_{sliceIdx}.jpeg`). Slices are centre-cropped to 224 x 224 and normalised with training-set statistics before entering the model.

| AD (patient 393968, slice 98) | NC (patient 1356105, slice 98) |
|---|---|
| ![AD input slice](readme_images/input_AD_393968_98.jpeg) | ![NC input slice](readme_images/input_NC_1356105_98.jpeg) |

#### Output of `train.py`

Example from `run_004` (`--model gfnet-xs --drop-path 0.3 --epochs 40`, best epoch 39, temperature T = 1.08). End of the console output:

```
TEST (subject level, n=450): acc 0.782  AUROC 0.848  macro-F1 0.777  sens 0.641  spec 0.921
TEST (slice level,   n=9000): acc 0.746  AUROC 0.813
ECE 0.066 -> 0.056 after temperature scaling;  AURC 0.151
  coverage 1.0: accuracy 0.782  (confidence >= 0.509)
  coverage 0.9: accuracy 0.805  (confidence >= 0.610)
  coverage 0.8: accuracy 0.822  (confidence >= 0.700)
  coverage 0.7: accuracy 0.838  (confidence >= 0.771)
  coverage 0.6: accuracy 0.867  (confidence >= 0.829)
  coverage 0.5: accuracy 0.871  (confidence >= 0.867)
```

Training curves (redrawn every epoch): loss, accuracy against the 0.8 target, validation AUROC and macro-F1, and the learning-rate schedule.

![Training curves](readme_images/train_curves.png)

Test results: ROC curves, calibration (reliability) before and after temperature scaling, risk-coverage curve for referral, and the patient-level confusion matrix.

![Test results](readme_images/test_results.png)

#### Output of `predict.py`

Example from `run_002` (`predict_001`). Each test patient gets P(AD) (mean of its 20 slice probabilities, temperature-calibrated), a confidence `max(P(AD), P(NC))`, and a decision: AD, NC, or REFER when confidence is below the referral threshold chosen on validation data. Excerpt of `predictions.csv`:

| patient_id | label | p_ad | confidence | predicted | decision | correct | slice_std |
|---|---|---|---|---|---|---|---|
| 393968 | AD | 0.9424 | 0.9424 | AD | AD | 1 | 0.0051 |
| 1356105 | NC | 0.0542 | 0.9458 | NC | NC | 1 | 0.0036 |
| 1223029 | AD | 0.0548 | 0.9452 | NC | NC | 0 | 0.0018 |
| 1261605 | NC | 0.8883 | 0.8883 | AD | AD | 0 | 0.1075 |
| 1220298 | AD | 0.5012 | 0.5012 | AD | REFER | 1 | 0.3371 |

Representative patients (confident correct AD and NC, false negatives, false positives and referred cases), each with an evidence heatmap. The AD-vs-NC logit splits exactly into one contribution per 16 x 16 patch: red patches push towards AD, blue towards NC.

![Prediction examples](readme_images/predict_examples.png)

Confidence of correct vs incorrect test patients, with the referral threshold:

![Confidence distribution](readme_images/predict_confidence.png)

Learned global filters: mean magnitude |K| over channels for each block, across the 2D frequency grid (zero frequency centred vertically).

![Learned global filters](readme_images/global_filters.png)

## Dataset Pre-processing

The ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset used here is the course's pre-processed subset of T1-weighted brain MRI scans from Alzheimer's disease (AD) and cognitively normal (NC) participants, provided as 2D greyscale sagittal slices of 240 x 256 pixels, 20 slices per patient [1].

| AD (patient 393968, slice 98) | NC (patient 1356105, slice 98) |
|---|---|
| ![AD slice](readme_images/input_AD_393968_98.jpeg) | ![NC slice](readme_images/input_NC_1356105_98.jpeg) |

### Patient-level train / validation / test split

| Split | AD patients (slices) | NC patients (slices) |
|---|---|---|
| Train (80% of `train/`) | 416 (8,320) | 445 (8,900) |
| Validation (20% of `train/`) | 104 (2,080) | 111 (2,220) |
| Test (provided `test/`) | 223 (4,460) | 227 (4,540) |

- Patients are split, not slices: all 20 slices of a patient sit in one split, so the model is never tested on a person it trained on. Slice-level splitting inflates accuracy through data leakage [2]. A check in `dataset.py` fails if any patient ID appears in two splits.
- The validation split is stratified by class, so it keeps the training AD/NC ratio, and it is fixed by `--seed`.
- Validation is used for early stopping, model selection, temperature scaling and choosing the referral threshold; the provided test set is used once, for final evaluation only.
- Caveat: patient identity comes from the filename prefix. If a prefix identifies a scan rather than a person, repeat visits of one person could be split across train and validation, which would make validation optimistic.

### Intensity normalisation

- Pixels are scaled to [0, 1], then standardised with the mean and standard deviation of the training split only, so no validation or test information leaks into preprocessing.
- The same statistics are saved in the checkpoint and reused at inference.

### Cropping to 224 x 224

- Training uses a random crop, evaluation a centre crop. The removed margin (16 rows, 32 columns) is background.
- 224 is the input size of the original GFNet [3] and a multiple of the 16 x 16 patch size, giving the 14 x 14 token grid the global filters operate on.

### Data augmentation (training only)

- Small random rotations (up to 10 degrees), shifts (up to 5%), scaling (0.95 to 1.05) and brightness/contrast jitter (up to 10%) mimic natural variation in head position and scanner intensity, reducing overfitting.
- No horizontal flips: Alzheimer's-related atrophy can be asymmetric, so mirroring could change the anatomy being classified.
- No Mixup or CutMix (used in the paper's ImageNet recipe [3]): blended images and labels would distort the softmax confidence that the referral analysis relies on.

**References**

1. Alzheimer's Disease Neuroimaging Initiative (ADNI). https://adni.loni.usc.edu
2. E. Yagis et al., "Effect of data leakage in brain MRI classification using 2D convolutional neural networks," *Scientific Reports*, vol. 11, 22544, 2021.
3. Y. Rao, W. Zhao, Z. Zhu, J. Lu, J. Zhou, "Global Filter Networks for Image Classification," *NeurIPS*, 2021. arXiv:2107.00645

## Core Investigation

Metrics and benchmarks to consider: 

- Model architecture's ability to reliably quantify uncertainty on ambiguous boundary cases
  - Run inference and examine a test case where the outcome is ambiguous, and then compare the classification probability after using softmax.
- Comparison of performance to a simple baseline
  - Comparison of the Top-1 accuracy and Top-5 accuracy percentages.
  - Compare performance to a generic CNN and MLP baseline
  - Consider the number of parameters and FLOPs required for the different models
- Correlation of model's predicted confidence score with diagnotic correctness
  - Comparison of the diagnostic label of the true data versus what the model predicted.
  - Check for false negative and false positive cases, and measure the degree of incorrectness.

## Engineering Reccomendation

## Establish empirical confidence threshold or reject option rule that automates ambiguous cases to human specialists.

- ## Whether the added complexity and computational cost justifies operational advantage over the baseline.