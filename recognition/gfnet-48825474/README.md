# GFNet for classification of Alzheimer's on the ADNI dataset

## About the GFNet model

A title, a description of the algorithm and the problem that it solves (approximately a paragraph), how it works in a paragraph, and a figure/visualisation.



## Quick Start

Dependencies, address reproducibility of results

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

Example inputs, outputs, and visual plots of your algorithm.

## Dataset Pre-processing 

Description of any specific pre-processing used, with references if any. Justify your training, validation, and testing splits of the data.





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