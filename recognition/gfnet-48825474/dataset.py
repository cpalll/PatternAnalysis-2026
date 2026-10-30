"""
dataset.py - ADNI AD vs NC data pipeline for GFNet.

Expected layout (course copy at /home/groups/comp3710/ADNI/AD_NC):

    AD_NC/
      train/AD/{patientID}_{sliceIdx}.jpeg
      train/NC/...
      test/AD/...
      test/NC/...

Images are 2D sagittal T1 slices, greyscale, 240 (H) x 256 (W), ~20 slices per patient.

Pipeline:
    1. Index every slice with its label and patient ID.
    2. Patient-level, class-stratified train/val split carved out of train/.
       test/ is never touched until final evaluation.
    3. Intensity normalisation using mean/std computed on the TRAIN split only.
    4. Spatial: centre crop (eval) or random crop + light affine (train) to 224 x 224.
    5. Each sample returns (image, label, patient_id) so predictions can be
       aggregated per subject for calibration and human-referral routing.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

CLASSES = {"NC": 0, "AD": 1}          # positive class = AD
_PID_RE = re.compile(r"^(\d+)_(\d+)\.jpe?g$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# 1. Indexing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    pid: str
    slice_idx: int


def index_split(root: str | Path, split: str) -> list[Sample]:
    """List every slice in root/split/{AD,NC} with label and patient ID."""
    samples = []
    for cls, label in CLASSES.items():
        folder = Path(root, split, cls)
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for p in sorted(folder.iterdir()):
            m = _PID_RE.match(p.name)
            if m:
                samples.append(Sample(p, label, m.group(1), int(m.group(2))))
    if not samples:
        raise RuntimeError(f"No images found under {Path(root, split)}")
    return samples


# --------------------------------------------------------------------------- #
# 2. Patient-level split
# --------------------------------------------------------------------------- #
def patient_split(samples: list[Sample], val_frac: float = 0.2, seed: int = 42):
    """Split slices into train/val so that every patient lives in exactly one
    split. Stratified by class so both splits keep the AD/NC ratio."""
    by_patient: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_patient[s.pid].append(s)

    # a patient must have a single label
    for pid, ss in by_patient.items():
        if len({s.label for s in ss}) != 1:
            raise ValueError(f"Patient {pid} appears in both AD and NC")

    rng = random.Random(seed)
    train, val = [], []
    for label in CLASSES.values():
        pids = sorted(p for p, ss in by_patient.items() if ss[0].label == label)
        rng.shuffle(pids)
        n_val = round(len(pids) * val_frac)
        for i, pid in enumerate(pids):
            (val if i < n_val else train).extend(by_patient[pid])
    return train, val


def assert_no_leakage(**splits: list[Sample]) -> None:
    """Raise if any patient ID appears in more than one split."""
    names = list(splits)
    pids = {n: {s.pid for s in splits[n]} for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = pids[a] & pids[b]
            if overlap:
                raise AssertionError(
                    f"Patient leakage between {a} and {b}: {sorted(overlap)[:5]}...")


def summarise(name: str, samples: list[Sample]) -> str:
    out = [name]
    for cls, label in CLASSES.items():
        ss = [s for s in samples if s.label == label]
        out.append(f"{cls}: {len({s.pid for s in ss})} patients / {len(ss)} slices")
    return " | ".join(out)


# --------------------------------------------------------------------------- #
# 3. Intensity statistics (train split only)
# --------------------------------------------------------------------------- #
def load_grey(path: Path) -> np.ndarray:
    """Load a slice as float32 in [0, 1], shape (H, W)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def compute_mean_std(samples: list[Sample], max_images: int = 2000,
                     seed: int = 0) -> tuple[float, float]:
    """Pixel mean/std over (a random subset of) the training slices."""
    rng = random.Random(seed)
    subset = samples if len(samples) <= max_images else rng.sample(samples, max_images)
    total, total_sq, n = 0.0, 0.0, 0
    for s in subset:
        x = load_grey(s.path).astype(np.float64)
        total += x.sum()
        total_sq += (x ** 2).sum()
        n += x.size
    mean = total / n
    std = (total_sq / n - mean ** 2) ** 0.5
    return float(mean), float(std)


# --------------------------------------------------------------------------- #
# 4. Transforms
# --------------------------------------------------------------------------- #
def build_transforms(mean: float, std: float, img_size: int = 224, train: bool = False):
    """Input: uint8 tensor (1, 240, 256). Output: float tensor (1, img_size, img_size).

    No horizontal flips (atrophy can be lateralised) and no Mixup/CutMix
    (they blur the ambiguity we want the softmax confidence to measure).
    """
    if train:
        spatial = [
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.RandomCrop(img_size),
        ]
        intensity = [T.ColorJitter(brightness=0.1, contrast=0.1)]
    else:
        spatial = [T.CenterCrop(img_size)]
        intensity = []
    return T.Compose([
        *spatial,
        *intensity,
        T.ToDtype(torch.float32, scale=True),     # uint8 [0,255] -> float [0,1]
        T.Normalize(mean=[mean], std=[std]),
    ])


# --------------------------------------------------------------------------- #
# 5. Dataset
# --------------------------------------------------------------------------- #
class ADNIDataset(Dataset):
    """Slice-level dataset returning (image, label, patient_id).

    cache=True decodes every JPEG once into RAM as uint8
    (~61 KB per slice, ~1.3 GB for the full train set), which removes the
    JPEG-decoding bottleneck on small machines.
    """

    def __init__(self, samples: list[Sample], transform=None, cache: bool = False):
        self.samples = samples
        self.transform = transform
        self._cache = None
        if cache:
            self._cache = [self._read(s.path) for s in samples]

    @staticmethod
    def _read(path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"), dtype=np.uint8)
        return torch.from_numpy(arr.copy()).unsqueeze(0)        # (1, H, W) uint8

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = self._cache[idx] if self._cache is not None else self._read(s.path)
        if self.transform is not None:
            x = self.transform(x)
        return x, s.label, s.pid


# --------------------------------------------------------------------------- #
# 6. DataLoaders
# --------------------------------------------------------------------------- #
def build_dataloaders(root: str | Path, batch_size: int = 64, img_size: int = 224,
                      val_frac: float = 0.2, num_workers: int = 4, seed: int = 42,
                      cache: bool = False, verbose: bool = True):
    """Return (train_loader, val_loader, test_loader, stats).

    stats = {"mean": float, "std": float} - save these with the checkpoint so
    inference uses identical normalisation.
    """
    train_all = index_split(root, "train")
    test = index_split(root, "test")
    train, val = patient_split(train_all, val_frac=val_frac, seed=seed)
    assert_no_leakage(train=train, val=val, test=test)

    mean, std = compute_mean_std(train)
    if verbose:
        for name, ss in (("train", train), ("val", val), ("test", test)):
            print(summarise(name, ss))
        print(f"train-split intensity mean={mean:.4f} std={std:.4f}")

    tf_train = build_transforms(mean, std, img_size, train=True)
    tf_eval = build_transforms(mean, std, img_size, train=False)

    ds_train = ADNIDataset(train, tf_train, cache=cache)
    ds_val = ADNIDataset(val, tf_eval, cache=cache)
    ds_test = ADNIDataset(test, tf_eval, cache=cache)

    g = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(),
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(ds_train, shuffle=True, drop_last=True, generator=g, **common)
    val_loader = DataLoader(ds_val, shuffle=False, **common)
    test_loader = DataLoader(ds_test, shuffle=False, **common)
    return train_loader, val_loader, test_loader, {"mean": mean, "std": std}


# --------------------------------------------------------------------------- #
# Subject-level aggregation helper (used for calibration / referral)
# --------------------------------------------------------------------------- #
def aggregate_by_patient(probs_ad: torch.Tensor, labels: torch.Tensor, pids: list[str]):
    """Average slice-level P(AD) per patient.

    Returns dict pid -> {"p_ad": mean prob, "std": spread across slices,
                         "label": int, "n_slices": int}
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    lab: dict[str, int] = {}
    for p, y, pid in zip(probs_ad.tolist(), labels.tolist(), pids):
        buckets[pid].append(p)
        lab[pid] = int(y)
    return {
        pid: {"p_ad": float(np.mean(v)), "std": float(np.std(v)),
              "label": lab[pid], "n_slices": len(v)}
        for pid, v in buckets.items()
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sanity-check the ADNI pipeline")
    ap.add_argument("--root", default="/home/groups/comp3710/ADNI/AD_NC")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    tr, va, te, stats = build_dataloaders(args.root, batch_size=args.batch_size,
                                          num_workers=args.workers)
    x, y, pid = next(iter(tr))
    print(f"batch x={tuple(x.shape)} dtype={x.dtype} "
          f"range=[{x.min():.2f}, {x.max():.2f}] y={y.tolist()} pid[0]={pid[0]}")
