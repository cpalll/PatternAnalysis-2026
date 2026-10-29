"""
train.py - train, validate, test and save a GFNet for ADNI AD vs NC classification.

Workflow
    1. Build patient-level train / val / test loaders            (dataset.py)
    2. Build a GFNet variant, optionally from ImageNet weights    (modules.py)
    3. Train with AdamW + linear warmup + cosine decay (DeiT/GFNet recipe),
       mixed precision on CUDA, validating every epoch at slice and subject level
    4. Keep the best checkpoint (by validation subject-level AUROC by default),
       stop early if validation stops improving
    5. Fit a softmax temperature on the validation set (calibration)
    6. Evaluate once on the held-out test set: per-class precision/recall,
       macro-F1, AUROC, ECE, risk-coverage (selective classification), and
       compute cost (params, peak VRAM, latency)
    7. Save checkpoint, history, metrics, per-patient test predictions and plots

Outputs (in a new numbered folder <--out-dir>/run_NNN, e.g. runs/run_001)
    best.pt / last.pt          checkpoints (best.pt also stores the fitted temperature)
    history.csv                per-epoch metrics
    training_curves.png        redrawn after every epoch
    test_results.png           ROC, reliability, risk-coverage, confusion matrix
    test_metrics.json          all final numbers
    test_predictions.csv       per-patient probabilities for failure-case analysis
    profile.md                 resource profile: params, FLOPs, peak VRAM, latency, throughput

Example
    python train.py --root /home/groups/comp3710/ADNI/AD_NC --model gfnet-xs \
        --epochs 60 --batch-size 64 --cache        # -> runs/run_001, runs/run_002, ...
"""

import argparse
import csv
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset import aggregate_by_patient, build_dataloaders  # noqa: E402
from modules import MODELS, GlobalFilter, adapt_pretrained_state_dict, build_model  # noqa: E402


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def get_args(argv=None):
    p = argparse.ArgumentParser(description="Train GFNet on ADNI AD vs NC")
    # data
    p.add_argument("--root", default="/home/groups/comp3710/ADNI/AD_NC")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", action="store_true", help="decode all images into RAM once")
    # model
    p.add_argument("--model", default="gfnet-xs", choices=sorted(MODELS))
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--pretrained", default=None, help="official GFNet ImageNet checkpoint (.pth)")
    # optimisation (GFNet/DeiT recipe, scaled down for a small dataset)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=None,
                   help="peak LR; default 1e-3 * batch/512 (scratch) or 2.5e-4 * batch/512 (pretrained)")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=None)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    # selection / stopping
    p.add_argument("--select", default="subj_auc", choices=["subj_auc", "subj_acc", "loss"],
                   help="validation metric used to pick the best checkpoint")
    p.add_argument("--patience", type=int, default=15, help="early-stopping patience (epochs)")
    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs",
                   help="parent folder; each run is saved to a new <out-dir>/run_NNN")
    p.add_argument("--limit-batches", type=int, default=0,
                   help="debug: cap batches per epoch (0 = no limit)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def next_run_dir(parent, prefix: str) -> Path:
    """Create and return the next numbered folder <parent>/<prefix>_NNN
    (run_001, run_002, ...) so earlier results are never overwritten."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for d in parent.iterdir()
            if d.is_dir() and (m := re.fullmatch(rf"{prefix}_(\d+)", d.name))]
    n = max(nums, default=0) + 1
    while True:
        path = parent / f"{prefix}_{n:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            n += 1


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def param_groups(model: nn.Module, weight_decay: float):
    """AdamW groups as in timm/DeiT: no decay on biases, norms, LayerScale gammas
    and names in model.no_weight_decay() (pos_embed). Global filters ARE decayed."""
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        if prm.ndim <= 1 or name.endswith(".bias") or name in skip:
            no_decay.append(prm)
        else:
            decay.append(prm)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(total_steps: int, warmup_steps: int, base_lr: float, min_lr: float):
    """Per-step LR multiplier: linear warmup then cosine decay to min_lr."""
    floor = min_lr / base_lr

    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


# --------------------------------------------------------------------------- #
# Metrics (pure torch)
# --------------------------------------------------------------------------- #
def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties averaged)."""
    scores, labels = scores.double().flatten(), labels.flatten()
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    _, inv, counts = torch.unique_consecutive(scores[order], return_inverse=True,
                                              return_counts=True)
    end = counts.cumsum(0).double()
    avg_rank = end - (counts.double() - 1) / 2
    ranks = torch.empty_like(scores)
    ranks[order] = avg_rank[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(scores: torch.Tensor, labels: torch.Tensor):
    """Returns (fpr, tpr) tensors, thresholds swept from high to low."""
    order = scores.argsort(descending=True)
    y = labels[order].double()
    tps = torch.cat([torch.zeros(1, dtype=torch.float64), y.cumsum(0)])
    fps = torch.cat([torch.zeros(1, dtype=torch.float64), (1 - y).cumsum(0)])
    return fps / max(1.0, fps[-1].item()), tps / max(1.0, tps[-1].item())


def classification_report(p_ad: torch.Tensor, labels: torch.Tensor, thr: float = 0.5) -> dict:
    """Accuracy, per-class precision/recall/F1, macro-F1, AUROC, confusion matrix.
    Positive class = AD (1)."""
    pred = (p_ad >= thr).long()
    labels = labels.long()
    cm = torch.zeros(2, 2, dtype=torch.long)       # rows = true, cols = predicted
    for t, q in zip(labels.tolist(), pred.tolist()):
        cm[t, q] += 1
    out = {"n": int(labels.numel()), "accuracy": float((pred == labels).double().mean()),
           "auroc": auroc(p_ad, labels), "confusion": cm.tolist()}
    f1s = []
    for c, name in ((0, "NC"), (1, "AD")):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = prec, rec, f1
        f1s.append(f1)
    out["macro_f1"] = sum(f1s) / 2
    out["sensitivity"] = out["recall_AD"]          # clinical names
    out["specificity"] = out["recall_NC"]
    return out


def calibration(p_ad: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> dict:
    """Top-label Expected Calibration Error plus per-bin data for a reliability diagram."""
    conf = torch.maximum(p_ad, 1 - p_ad).double()
    correct = ((p_ad >= 0.5).long() == labels.long()).double()
    edges = torch.linspace(0.5, 1.0, n_bins + 1, dtype=torch.float64)
    ece, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.any():
            acc, cf, w = correct[m].mean().item(), conf[m].mean().item(), m.double().mean().item()
            ece += w * abs(acc - cf)
            bins.append({"lo": lo.item(), "hi": hi.item(), "acc": acc, "conf": cf, "n": int(m.sum())})
    return {"ece": ece, "bins": bins}


def risk_coverage(p_ad: torch.Tensor, labels: torch.Tensor) -> dict:
    """Selective classification: accept the most confident cases first.

    coverage[k] = fraction accepted, risk[k] = error rate among accepted.
    AURC = mean risk over all coverage levels (lower is better).
    """
    conf = torch.maximum(p_ad, 1 - p_ad)
    err = ((p_ad >= 0.5).long() != labels.long()).double()
    order = conf.argsort(descending=True)
    err, conf = err[order], conf[order]
    k = torch.arange(1, len(err) + 1, dtype=torch.float64)
    risk = err.cumsum(0) / k
    coverage = k / len(err)
    at = {}
    for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        idx = max(0, int(math.ceil(c * len(err))) - 1)
        at[f"{c:.1f}"] = {"accuracy": 1 - risk[idx].item(), "threshold": conf[idx].item()}
    return {"coverage": coverage, "risk": risk, "aurc": risk.mean().item(), "at_coverage": at}


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling (Guo et al., 2017): find T > 0 minimising validation NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    logits, labels = logits.float(), labels.long()

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def subject_level(p_ad: torch.Tensor, labels: torch.Tensor, pids: list):
    """Average slice P(AD) per patient -> (p_ad, labels, pid list, slice std)."""
    agg = aggregate_by_patient(p_ad, labels, pids)
    keys = sorted(agg)
    return (torch.tensor([agg[k]["p_ad"] for k in keys], dtype=torch.float64),
            torch.tensor([agg[k]["label"] for k in keys]),
            keys,
            torch.tensor([agg[k]["std"] for k in keys], dtype=torch.float64))


# --------------------------------------------------------------------------- #
# Train / evaluate loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device,
                    use_amp: bool, clip_grad=None, limit: int = 0) -> dict:
    model.train()
    tot_loss, tot_correct, tot_n = 0.0, 0, 0
    for i, (x, y, _) in enumerate(loader):
        if limit and i >= limit:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if clip_grad:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if not math.isfinite(loss.item()):
            raise RuntimeError("Loss is NaN/inf - lower the LR or disable AMP")
        tot_loss += loss.item() * y.size(0)
        tot_correct += (logits.argmax(1) == y).sum().item()
        tot_n += y.size(0)
    return {"loss": tot_loss / tot_n, "acc": tot_correct / tot_n,
            "lr": optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def predict(model, loader, device, use_amp: bool, limit: int = 0):
    """Run the model over a loader -> (logits[float32, N x 2], labels[N], pids[list])."""
    model.eval()
    all_logits, all_labels, all_pids = [], [], []
    for i, (x, y, pid) in enumerate(loader):
        if limit and i >= limit:
            break
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x.to(device, non_blocking=True))
        all_logits.append(logits.float().cpu())
        all_labels.append(y)
        all_pids.extend(pid)
    return torch.cat(all_logits), torch.cat(all_labels), all_pids


def evaluate(logits, labels, pids, criterion=None, temperature: float = 1.0) -> dict:
    """Slice- and subject-level metrics from stored logits."""
    p_ad = (logits / temperature).softmax(1)[:, 1].double()
    out = {"slice": classification_report(p_ad, labels)}
    if criterion is not None:
        out["loss"] = criterion(logits, labels.long()).item()
    sp, sy, _, _ = subject_level(p_ad, labels, pids)
    out["subject"] = classification_report(sp, sy)
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"        # categorical slots 1-3 (fixed order)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
    "lines.linewidth": 1.6, "font.size": 9,
})


def plot_training(history: list, path: Path, model_name: str) -> None:
    """2x2 panel: loss, accuracy, validation AUROC / macro-F1, learning rate."""
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    a = ax[0, 0]
    a.plot(ep, [h["train_loss"] for h in history], color=C1, label="Train")
    a.plot(ep, [h["val_loss"] for h in history], color=C2, label="Validation")
    a.set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    a.legend()

    a = ax[0, 1]
    a.plot(ep, [h["train_acc"] for h in history], color=C1, label="Train (slice)")
    a.plot(ep, [h["val_acc"] for h in history], color=C2, label="Validation (slice)")
    a.plot(ep, [h["val_subj_acc"] for h in history], color=C3, label="Validation (subject)")
    a.axhline(0.8, color=INK2, lw=0.8, ls="--")
    a.text(ep[0], 0.805, "0.8 target", color=INK2, fontsize=7, va="bottom")
    a.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    a.legend(loc="lower right")

    a = ax[1, 0]
    a.plot(ep, [h["val_auc"] for h in history], color=C1, label="AUROC (slice)")
    a.plot(ep, [h["val_subj_auc"] for h in history], color=C2, label="AUROC (subject)")
    a.plot(ep, [h["val_macro_f1"] for h in history], color=C3, label="Macro-F1 (subject)")
    a.set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    a.legend(loc="lower right")

    a = ax[1, 1]
    a.plot(ep, [h["lr"] for h in history], color=C1)
    a.set(title="Learning rate (end of epoch)", xlabel="Epoch", ylabel="LR")
    a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for a in ax.flat:
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    fig.suptitle(f"{model_name} training on ADNI (AD vs NC)", color=INK, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_test(res: dict, path: Path, model_name: str) -> None:
    """ROC, reliability diagram, risk-coverage and confusion matrix for the test set."""
    fig, ax = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)

    a = ax[0, 0]
    for key, color, label in (("slice", C1, "Slice"), ("subject", C2, "Subject")):
        fpr, tpr = res["roc"][key]
        a.plot(fpr, tpr, color=color, label=f"{label} (AUROC {res['metrics'][key]['auroc']:.3f})")
    a.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    a.set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate",
          xlim=(0, 1), ylim=(0, 1.01))
    a.legend(loc="lower right")

    a = ax[0, 1]
    for key, color, label in (("raw", C1, "Uncalibrated"), ("cal", C2, "Temperature-scaled")):
        bins = res["calibration"][key]["bins"]
        ece = res["calibration"][key]["ece"]
        a.plot([b["conf"] for b in bins], [b["acc"] for b in bins], color=color, marker="o",
               ms=4, label=f"{label} (ECE {ece:.3f})")
    a.plot([0.5, 1], [0.5, 1], color=INK2, lw=0.8, ls="--")
    a.set(title="Reliability (test, subject level)", xlabel="Mean confidence",
          ylabel="Accuracy", xlim=(0.5, 1), ylim=(0.0, 1.02))
    a.legend(loc="lower right")

    a = ax[1, 0]
    rc = res["risk_coverage"]
    a.plot(rc["coverage"], rc["risk"], color=C1, label=f"GFNet (AURC {rc['aurc']:.3f})")
    a.axhline(rc["risk"][-1].item(), color=INK2, lw=0.8, ls="--", label="Error with no referral")
    a.set(title="Risk-coverage (test, subject level)", xlabel="Coverage (fraction auto-decided)",
          ylabel="Error rate on accepted cases", xlim=(0, 1), ylim=(0, None))
    a.legend(loc="upper left")

    a = ax[1, 1]
    cm = torch.tensor(res["metrics"]["subject"]["confusion"])
    a.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#1c5cab"]))
    a.grid(False)
    for i in range(2):
        for j in range(2):
            v, row = cm[i, j].item(), cm[i].sum().item()
            a.text(j, i, f"{v}\n({v / row:.0%})" if row else f"{v}", ha="center", va="center",
                   color="#ffffff" if v > cm.max().item() / 2 else INK, fontsize=10)
    a.set_xticks([0, 1], ["NC", "AD"])
    a.set_yticks([0, 1], ["NC", "AD"])
    a.set(title="Confusion matrix (test, subject level)", xlabel="Predicted", ylabel="True")

    fig.suptitle(f"{model_name} test results (T = {res['temperature']:.2f})", color=INK, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Compute profiling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def profile_latency(model, device, img_size: int, batch_size: int, use_amp: bool,
                    reps: int = 20) -> dict:
    """Mean forward-pass time (ms) for one slice and for one full batch."""
    model.eval()
    out = {}
    for name, bs in (("latency_ms_per_slice_bs1", 1), (f"latency_ms_per_batch_bs{batch_size}", batch_size)):
        x = torch.randn(bs, 1, img_size, img_size, device=device)
        for _ in range(3):                                   # warm-up
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        out[name] = (time.perf_counter() - t0) / reps * 1000
    out["latency_batch_size"] = batch_size
    out["throughput_slices_per_s"] = batch_size / (out[f"latency_ms_per_batch_bs{batch_size}"] / 1000)
    return out


@torch.no_grad()
def count_flops(model, device, img_size: int) -> dict:
    """FLOPs for one 1 x img_size x img_size slice.

    gflops_paper: PyTorch's FLOP counter / 2, i.e. 1 multiply-add = 1 FLOP, the
                  convention used in the GFNet paper's tables (comparable to them).
    fft_gflops_est: analytic estimate for the FFTs and filter products, which the
                  counter does not see (rfft2 + irfft2 ~ 2 x 2.5 N log2 N real
                  FLOPs per channel, plus 6 FLOPs per complex product).
    """
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    x = torch.randn(1, 1, img_size, img_size, device=device)
    with FlopCounterMode(display=False) as fc:
        model(x)
    fft = 0.0
    for m in model.modules():
        if isinstance(m, GlobalFilter):
            n, ch = m.h * m.w, m.complex_weight.shape[2]
            fft += ch * (2 * 2.5 * n * math.log2(n) + 6 * m.h * m.freq_w)
    return {"gflops_paper": fc.get_total_flops() / 2e9, "fft_gflops_est": fft / 1e9}


def environment(device, use_amp: bool) -> dict:
    """Hardware/software the measurements were taken on."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem = torch.cuda.get_device_properties(device).total_memory / 2**30
        dev = f"{name} ({mem:.0f} GB)"
    else:
        dev = device.type.upper()
    return {"device": dev, "mixed_precision": use_amp, "torch": torch.__version__,
            "cuda": torch.version.cuda if device.type == "cuda" else None}


def write_profile(cost: dict, path: Path, model_name: str) -> None:
    """Resource profile as a Markdown table (pasteable into the README)."""
    def gb(v, batch):
        return f"{v:.2f} GB (batch {batch})" if v is not None else "n/a (CUDA only)"
    bs = cost["latency_batch_size"]
    rows = [
        ("Model", model_name),
        ("Parameters", f"{cost['params'] / 1e6:.2f} M ({cost['params']:,})"),
        ("FLOPs per slice (1 MAC = 1 FLOP, as in the paper)", f"{cost['gflops_paper']:.2f} G"),
        ("FFT FLOPs per slice (analytic estimate, not included above)", f"{cost['fft_gflops_est']:.3f} G"),
        ("Peak GPU VRAM, training", gb(cost["peak_train_vram_gb"], cost["train_batch_size"])),
        ("Peak GPU VRAM, inference", gb(cost["peak_test_vram_gb"], bs)),
        ("Inference latency, 1 slice", f"{cost['latency_ms_per_slice_bs1']:.2f} ms"),
        (f"Inference latency, batch of {bs}", f"{cost[f'latency_ms_per_batch_bs{bs}']:.2f} ms"),
        ("Inference throughput", f"{cost['throughput_slices_per_s']:,.0f} slices/s"),
        ("Inference time per patient (20 slices, batched)", f"{20 * 1000 / cost['throughput_slices_per_s']:.1f} ms"),
        ("Mean epoch time (train + validation)", f"{cost['mean_epoch_time_s']:.1f} s"),
        ("Total training time", f"{cost['total_train_time_s'] / 60:.1f} min ({cost['epochs_run']} epochs)"),
        ("Device", cost["device"]),
        ("Mixed precision (FP16)", "yes" if cost["mixed_precision"] else "no"),
        ("PyTorch / CUDA", f"{cost['torch']} / {cost['cuda'] or '-'}"),
    ]
    lines = [f"# Resource profile: {model_name}", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = get_args(argv)
    set_seed(args.seed)
    device = get_device()
    use_amp = device.type == "cuda" and not args.no_amp
    out_dir = next_run_dir(args.out_dir, "run")
    print(f"device={device}  amp={use_amp}  out={out_dir}")

    # ---- data ----
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        args.root, batch_size=args.batch_size, img_size=args.img_size,
        val_frac=args.val_frac, num_workers=args.workers, seed=args.seed, cache=args.cache)

    # ---- model ----
    model_kwargs = dict(img_size=args.img_size, in_chans=1, num_classes=2,
                        drop_path_rate=args.drop_path, drop_rate=args.drop)
    model = build_model(args.model, **model_kwargs)
    if args.pretrained:
        sd = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        res = model.load_state_dict(adapt_pretrained_state_dict(sd, model), strict=False)
        print(f"loaded {args.pretrained}  missing={res.missing_keys}")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params / 1e6:.2f}M parameters")

    # ---- optimisation ----
    base = 2.5e-4 if args.pretrained else 1e-3
    lr = args.lr or base * args.batch_size / 512
    steps_per_epoch = len(train_loader) if not args.limit_batches else min(len(train_loader), args.limit_batches)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_steps, steps_per_epoch * args.warmup_epochs, lr, args.min_lr))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    print(f"lr={lr:.2e}  steps/epoch={steps_per_epoch}  epochs={args.epochs}")

    def checkpoint(epoch, metrics, temperature=1.0):
        return {"model": model.state_dict(), "model_name": args.model, "model_kwargs": model_kwargs,
                "data_stats": stats, "args": vars(args), "epoch": epoch,
                "val_metrics": metrics, "temperature": temperature}

    # ---- training loop ----
    history, best, best_epoch, bad_epochs = [], None, 0, 0
    sign = -1 if args.select == "loss" else 1                  # maximise score, minimise loss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, train_criterion, optimizer, scheduler, scaler,
                             device, use_amp, args.clip_grad, args.limit_batches)
        logits, labels, pids = predict(model, val_loader, device, use_amp, args.limit_batches)
        va = evaluate(logits, labels, pids, eval_criterion)
        row = {"epoch": epoch, "lr": tr["lr"], "train_loss": tr["loss"], "train_acc": tr["acc"],
               "val_loss": va["loss"], "val_acc": va["slice"]["accuracy"],
               "val_auc": va["slice"]["auroc"], "val_subj_acc": va["subject"]["accuracy"],
               "val_subj_auc": va["subject"]["auroc"], "val_macro_f1": va["subject"]["macro_f1"],
               "epoch_time_s": time.time() - t0}
        history.append(row)

        score = {"subj_auc": row["val_subj_auc"], "subj_acc": row["val_subj_acc"],
                 "loss": row["val_loss"]}[args.select]
        improved = best is None or sign * score > sign * best
        if improved:
            best, best_epoch, bad_epochs = score, epoch, 0
            torch.save(checkpoint(epoch, va), out_dir / "best.pt")
        else:
            bad_epochs += 1
        torch.save(checkpoint(epoch, va), out_dir / "last.pt")

        print(f"[{epoch:3d}/{args.epochs}] lr {tr['lr']:.2e} | train loss {tr['loss']:.4f} "
              f"acc {tr['acc']:.3f} | val loss {va['loss']:.4f} acc {row['val_acc']:.3f} "
              f"subj acc {row['val_subj_acc']:.3f} subj AUROC {row['val_subj_auc']:.3f} | "
              f"{row['epoch_time_s']:.0f}s{'  *best' if improved else ''}")

        with open(out_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)
        plot_training(history, out_dir / "training_curves.png", args.model)

        if bad_epochs >= args.patience:
            print(f"early stopping: no improvement in {args.patience} epochs")
            break

    peak_train_vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    print(f"best epoch {best_epoch} ({args.select} = {best:.4f})")

    # ---- calibration on validation ----
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_logits, val_labels, val_pids = predict(model, val_loader, device, use_amp, args.limit_batches)
    T = fit_temperature(val_logits, val_labels)
    ckpt["temperature"] = T
    torch.save(ckpt, out_dir / "best.pt")
    print(f"fitted temperature T = {T:.3f}")

    # ---- test (held out, evaluated once) ----
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_logits, t_labels, t_pids = predict(model, test_loader, device, use_amp, args.limit_batches)
    peak_infer_vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None

    metrics = evaluate(t_logits, t_labels, t_pids, eval_criterion, temperature=T)
    raw = evaluate(t_logits, t_labels, t_pids, eval_criterion, temperature=1.0)

    p_raw = t_logits.softmax(1)[:, 1].double()
    p_cal = (t_logits / T).softmax(1)[:, 1].double()
    s_raw, s_y, s_pid, _ = subject_level(p_raw, t_labels, t_pids)
    s_cal, _, _, s_std = subject_level(p_cal, t_labels, t_pids)

    rc = risk_coverage(s_cal, s_y)
    results = {
        "temperature": T,
        "metrics": metrics,
        "roc": {"slice": roc_curve(p_cal, t_labels), "subject": roc_curve(s_cal, s_y)},
        "calibration": {"raw": calibration(s_raw, s_y), "cal": calibration(s_cal, s_y)},
        "risk_coverage": rc,
    }
    plot_test(results, out_dir / "test_results.png", args.model)

    total_time = sum(h["epoch_time_s"] for h in history)
    cost = {"params": n_params, "peak_train_vram_gb": peak_train_vram,
            "peak_test_vram_gb": peak_infer_vram, "train_batch_size": args.batch_size,
            "mean_epoch_time_s": total_time / len(history), "total_train_time_s": total_time,
            "epochs_run": len(history)}
    cost.update(profile_latency(model, device, args.img_size, args.batch_size, use_amp))
    cost.update(count_flops(model, device, args.img_size))
    cost.update(environment(device, use_amp))
    write_profile(cost, out_dir / "profile.md", args.model)

    summary = {
        "run_dir": str(out_dir), "model": args.model, "best_epoch": best_epoch, "selected_by": args.select,
        "temperature": T,
        "test": {"slice": metrics["slice"], "subject": metrics["subject"],
                 "loss": metrics["loss"], "uncalibrated_subject": raw["subject"]},
        "calibration_subject": {"ece_raw": results["calibration"]["raw"]["ece"],
                                "ece_temperature_scaled": results["calibration"]["cal"]["ece"]},
        "selective_subject": {"aurc": rc["aurc"], "at_coverage": rc["at_coverage"]},
        "cost": cost, "args": vars(args),
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "label", "p_ad_raw", "p_ad_calibrated", "confidence",
                    "predicted", "correct", "slice_std"])
        for pid, y, pr, pc, sd in zip(s_pid, s_y.tolist(), s_raw.tolist(), s_cal.tolist(), s_std.tolist()):
            pred = int(pc >= 0.5)
            w.writerow([pid, y, f"{pr:.4f}", f"{pc:.4f}", f"{max(pc, 1 - pc):.4f}",
                        pred, int(pred == y), f"{sd:.4f}"])

    m = metrics["subject"]
    print(f"\nTEST (subject level, n={m['n']}): acc {m['accuracy']:.3f}  AUROC {m['auroc']:.3f}  "
          f"macro-F1 {m['macro_f1']:.3f}  sens {m['sensitivity']:.3f}  spec {m['specificity']:.3f}")
    print(f"TEST (slice level,   n={metrics['slice']['n']}): acc {metrics['slice']['accuracy']:.3f}  "
          f"AUROC {metrics['slice']['auroc']:.3f}")
    print(f"ECE {results['calibration']['raw']['ece']:.3f} -> {results['calibration']['cal']['ece']:.3f} "
          f"after temperature scaling;  AURC {rc['aurc']:.3f}")
    for c, v in rc["at_coverage"].items():
        print(f"  coverage {c}: accuracy {v['accuracy']:.3f}  (confidence >= {v['threshold']:.3f})")
    print(f"\n{(out_dir / 'profile.md').read_text()}")
    print(f"saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
