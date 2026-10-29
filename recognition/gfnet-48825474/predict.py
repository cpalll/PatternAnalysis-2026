"""
predict.py - example usage of a trained GFNet for ADNI AD vs NC classification.

What it does
    1. Loads the checkpoint saved by train.py (weights, model config,
       normalisation stats and the fitted softmax temperature T).
    2. Runs inference on every test patient: slice-level P(AD) is averaged into a
       patient-level probability, then temperature-calibrated.
    3. Applies a human-referral rule: the confidence threshold tau is chosen on
       the VALIDATION patients as the lowest confidence at which accuracy on the
       automatically decided cases still meets --target-acc. Test patients below
       tau are routed to a human examiner.
    4. Prints a summary and a per-patient table, and saves visualisations:
         examples.png            representative slices: confident AD / NC,
                                 false negatives / positives, referred cases,
                                 each with an evidence heatmap overlay
         confidence.png          confidence distribution of correct vs incorrect
                                 test patients with the referral threshold
         global_filters.png      learned global filter magnitude |K| per block
                                 (frequency response of the token mixer)
         predictions.csv         per-patient results and decisions

    Each call writes to a new numbered folder inside the run it evaluates,
    e.g. runs/run_003/predict_001, predict_002, ... (nothing is overwritten).

    It can also be pointed at arbitrary slices with --images (no labels needed).

Evidence heatmap
    GFNet's head is linear on the global average of the final (LayerNorm-ed)
    tokens, so the AD-vs-NC logit difference decomposes exactly into per-token
    contributions:

        z_AD - z_NC = sum_t  (w_AD - w_NC) . LN(x_t) / N  +  (b_AD - b_NC)

    Each 16x16 patch therefore gets a signed score (red = evidence for AD,
    blue = evidence for NC) that sums to the model's decision. This is the CAM
    idea applied to GFNet, with no gradients or approximations.

Examples
    python predict.py                                   # latest run's best.pt
    python predict.py --checkpoint runs/run_002/best.pt --target-acc 0.9
    python predict.py --images path/to/slices/
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from dataset import (ADNIDataset, Sample, build_transforms, index_split,  # noqa: E402
                     patient_split)
from modules import GlobalFilter, build_model  # noqa: E402

LABELS = {0: "NC", 1: "AD"}

# palette: categorical slots 1-2, diverging blue <-> red with neutral grey midpoint
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2 = "#2a78d6", "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list("nc_ad", ["#2a78d6", "#f0efec", "#e34948"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 9, "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
    "legend.fontsize": 8, "legend.labelcolor": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def get_args(argv=None):
    p = argparse.ArgumentParser(description="Run a trained GFNet on ADNI test cases")
    p.add_argument("--checkpoint", default=None,
                   help="default: best.pt of the latest run in --runs-dir")
    p.add_argument("--runs-dir", default="runs", help="parent folder of train.py runs")
    p.add_argument("--root", default=None, help="AD_NC folder (default: the one used in training)")
    p.add_argument("--images", nargs="*", default=None,
                   help="predict on these slice files / folders instead of the test set")
    p.add_argument("--target-acc", type=float, default=0.95,
                   help="required accuracy on automatically decided patients")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--show", type=int, default=15, help="patients to print in the table")
    p.add_argument("--out-dir", default=None,
                   help="parent folder for results (default: the checkpoint's run folder); "
                        "each call writes to a new <out-dir>/predict_NNN")
    return p.parse_args(argv)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def latest_checkpoint(runs_dir) -> Path:
    """best.pt of the highest-numbered run_NNN folder."""
    runs = sorted((d for d in Path(runs_dir).glob("run_*")
                   if re.fullmatch(r"run_\d+", d.name) and (d / "best.pt").exists()),
                  key=lambda d: int(d.name.split("_")[1]))
    if not runs:
        raise FileNotFoundError(f"No run_NNN/best.pt found in {runs_dir}; pass --checkpoint")
    return runs[-1] / "best.pt"


def load_model(path, device):
    """Rebuild the model from the checkpoint written by train.py."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def final_tokens(model, x):
    """Token features after the final LayerNorm, before pooling: (B, N, D).
    Works for both GFNet and GFNetPyramid."""
    if isinstance(model.patch_embed, torch.nn.ModuleList):           # pyramid
        for i, (embed, stage) in enumerate(zip(model.patch_embed, model.blocks)):
            x = embed(x)
            if i == 0:
                x = x + model.pos_embed
            x = stage(x)
    else:                                                            # isotropic
        x = model.patch_embed(x) + model.pos_embed
        for blk in model.blocks:
            x = blk(x)
    return model.norm(x)


def last_grid(model):
    """Token grid (h, w) of the last stage, read from its global filter."""
    filters = [m for m in model.modules() if isinstance(m, GlobalFilter)]
    return filters[-1].h, filters[-1].w


@torch.no_grad()
def run_inference(model, loader, device, temperature):
    """Return per-slice records: logits, calibrated P(AD) and the
    token-contribution map (h, w) for the AD-vs-NC logit difference."""
    w = model.head.weight                                    # (2, D)
    diff = (w[1] - w[0]).float()
    bias = float(model.head.bias[1] - model.head.bias[0])
    h, wg = last_grid(model)
    recs = []
    for x, y, pid in loader:
        tokens = final_tokens(model, x.to(device)).float()   # (B, N, D)
        logits = model.head(tokens.mean(1))                  # identical to model(x) in eval
        contrib = (tokens @ diff) / tokens.shape[1]          # (B, N); sums to logit diff - bias
        p_ad = (logits / temperature).softmax(1)[:, 1]
        for i in range(len(pid)):
            recs.append({"pid": pid[i], "label": int(y[i]), "logits": logits[i].cpu(),
                         "p_ad": float(p_ad[i]),
                         "map": contrib[i].reshape(h, wg).cpu()})
        # sanity check of the exact decomposition on the first batch
        if len(recs) == len(pid):
            err = (contrib.sum(1) + bias - (logits[:, 1] - logits[:, 0])).abs().max().item()
            assert err < 1e-3, f"heatmap decomposition error {err}"
    return recs


def per_patient(recs, samples):
    """Average slice probabilities per patient; keep the middle slice for display."""
    by = defaultdict(list)
    for r, s in zip(recs, samples):
        by[r["pid"]].append((s.slice_idx, r, s))
    out = []
    for pid, items in by.items():
        items.sort(key=lambda t: t[0])
        p = sum(r["p_ad"] for _, r, _ in items) / len(items)
        slice_p = [r["p_ad"] for _, r, _ in items]
        mid_idx, mid_rec, mid_s = items[len(items) // 2]
        out.append({"pid": pid, "label": items[0][1]["label"], "p_ad": p,
                    "confidence": max(p, 1 - p), "pred": int(p >= 0.5),
                    "slice_std": float(np.std(slice_p)), "n_slices": len(items),
                    "slice_p": slice_p, "mid_path": mid_s.path, "mid_idx": mid_idx,
                    "mid_map": mid_rec["map"], "mid_p": mid_rec["p_ad"]})
    return sorted(out, key=lambda r: r["pid"])


def choose_threshold(patients, target_acc):
    """Lowest confidence tau such that accuracy on patients with confidence >= tau
    is at least target_acc (maximises coverage). Chosen on validation data."""
    pts = sorted(patients, key=lambda r: -r["confidence"])
    correct = np.array([r["pred"] == r["label"] for r in pts], dtype=float)
    acc = np.cumsum(correct) / np.arange(1, len(pts) + 1)
    ok = np.where(acc >= target_acc)[0]
    if len(ok) == 0:
        return 1.0, 0.0, float("nan")         # target unreachable: refer everyone
    k = ok[-1]
    return float(pts[k]["confidence"]), (k + 1) / len(pts), float(acc[k])


def predict_split(model, samples, tf, args, device, T):
    loader = DataLoader(ADNIDataset(samples, tf), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers)
    return per_patient(run_inference(model, loader, device, T), samples)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def load_display(path, img_size):
    """Original slice, centre-cropped exactly like the model input, as float [0,1]."""
    from PIL import Image
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    H, W = img.shape
    top, left = (H - img_size) // 2, (W - img_size) // 2
    return img[top:top + img_size, left:left + img_size]


def overlay(ax, img, cmap_map, vmax):
    """Grey slice with a signed evidence map on top; opacity follows |evidence|."""
    up = F.interpolate(cmap_map[None, None], size=img.shape, mode="bilinear",
                       align_corners=False)[0, 0].numpy()
    norm = np.clip(up / vmax, -1, 1)
    rgba = DIVERGING((norm + 1) / 2)
    rgba[..., 3] = np.abs(norm) * 0.65
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.imshow(rgba)
    ax.set_xticks([]), ax.set_yticks([])


def pick_examples(patients, tau, per_group=2):
    """Representative cases for the figure."""
    auto = [r for r in patients if r["confidence"] >= tau]
    refer = sorted([r for r in patients if r["confidence"] < tau], key=lambda r: r["confidence"])
    correct = lambda r: r["pred"] == r["label"]  # noqa: E731
    by_conf = lambda rs: sorted(rs, key=lambda r: -r["confidence"])  # noqa: E731
    groups = [
        ("Confident AD (correct)", by_conf([r for r in auto if correct(r) and r["label"] == 1])),
        ("Confident NC (correct)", by_conf([r for r in auto if correct(r) and r["label"] == 0])),
        ("False negative (AD called NC)", by_conf([r for r in patients if r["label"] == 1 and r["pred"] == 0])),
        ("False positive (NC called AD)", by_conf([r for r in patients if r["label"] == 0 and r["pred"] == 1])),
        ("Referred to human (least confident)", refer),
    ]
    picked, seen = [], set()
    for g, rs in groups:
        rs = [r for r in rs if r["pid"] not in seen][:per_group]
        seen.update(r["pid"] for r in rs)
        if rs:
            picked.append((g, rs))
    return picked


def plot_examples(groups, tau, img_size, path, has_labels=True):
    n_rows = len(groups)
    n_cols = max(len(rs) for _, rs in groups)
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(4.4 * n_cols, 2.6 * n_rows),
                             constrained_layout=True, squeeze=False)
    all_maps = [r["mid_map"] for _, rs in groups for r in rs]
    vmax = max(float(torch.stack([m.abs().max() for m in all_maps]).max()), 1e-6)
    for row, (title, rs) in enumerate(groups):
        for col in range(n_cols):
            a_img, a_map = axes[row, 2 * col], axes[row, 2 * col + 1]
            if col >= len(rs):
                a_img.axis("off"), a_map.axis("off")
                continue
            r = rs[col]
            img = load_display(r["mid_path"], img_size)
            a_img.imshow(img, cmap="gray", vmin=0, vmax=1)
            a_img.set_xticks([]), a_img.set_yticks([])
            decision = LABELS[r["pred"]] if r["confidence"] >= tau else "REFER"
            truth = f"true {LABELS[r['label']]} | " if has_labels else ""
            a_img.set_title(f"{r['pid']} slice {r['mid_idx']}\n{truth}P(AD) {r['p_ad']:.2f} -> {decision}",
                            fontsize=8)
            overlay(a_map, img, r["mid_map"], vmax)
            a_map.set_title("evidence (red AD, blue NC)", fontsize=8, color=INK2)
        axes[row, 0].set_ylabel(title, fontsize=9, color=INK, rotation=90, labelpad=6)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.01)
    cb.set_label("contribution to logit(AD) - logit(NC)", color=INK2)
    fig.suptitle(f"GFNet predictions on test patients (referral threshold: confidence < {tau:.2f})",
                 color=INK, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confidence(patients, tau, path):
    conf_ok = [r["confidence"] for r in patients if r["pred"] == r["label"]]
    conf_bad = [r["confidence"] for r in patients if r["pred"] != r["label"]]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bins = np.linspace(0.5, 1.0, 21)
    ax.axvspan(0.5, tau, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.hist([conf_ok, conf_bad], bins=bins, stacked=True, color=[C1, C2],
            label=[f"Correct (n={len(conf_ok)})", f"Incorrect (n={len(conf_bad)})"],
            edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax.axvline(tau, color=INK, lw=1, ls="--", label=f"Referral threshold {tau:.2f}", zorder=3)
    ax.text(0.505, ax.get_ylim()[1] * 0.97, "referred to human", ha="left", va="top",
            color=INK2, fontsize=8)
    ax.text(min(tau + 0.005, 0.99), ax.get_ylim()[1] * 0.97, "auto-decided", ha="left", va="top",
            color=INK2, fontsize=8)
    ax.set(xlabel="Calibrated confidence  max(P(AD), P(NC))", ylabel="Patients", xlim=(0.5, 1.0))
    ax.set_title("Confidence of test patients: correct vs incorrect", pad=24)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_filters(model, path, max_blocks=12):
    """Mean |K| over channels for each block's global filter. Rows are vertical
    frequency (fft-shifted so 0 is centred), columns horizontal frequency 0..Nyquist."""
    filters = [m for m in model.modules() if isinstance(m, GlobalFilter)]
    idx = np.unique(np.linspace(0, len(filters) - 1, min(max_blocks, len(filters))).round().astype(int))
    n = len(idx)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 2.3 * rows),
                             constrained_layout=True, squeeze=False)
    for a in axes.flat:
        a.axis("off")
    for a, i in zip(axes.flat, idx):
        f = filters[i]
        mag = torch.view_as_complex(f.complex_weight.detach().float()).abs().mean(-1)  # (h, w//2+1)
        mag = torch.fft.fftshift(mag, dim=0).cpu().numpy()
        a.imshow(mag, cmap=SEQUENTIAL, aspect="auto")
        a.set_title(f"block {i + 1}  ({f.h}x{f.w})", fontsize=8)
        a.axis("on")
        a.set_xticks([]), a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    axes[0, 0].set_ylabel("vertical freq.\n(0 centred)", fontsize=8)
    axes[-1, 0].set_xlabel("horizontal freq. 0 -> Nyquist", fontsize=8)
    fig.suptitle("Learned global filters: mean |K| over channels (darker = stronger)",
                 color=INK, fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Custom images
# --------------------------------------------------------------------------- #
def collect_images(paths):
    """Files or folders -> Sample list (label -1). Slices named {pid}_{idx}.jpeg are
    grouped into patients; other files are treated as one-slice cases."""
    files = []
    for p in map(Path, paths):
        files += sorted(q for q in p.rglob("*") if q.suffix.lower() in (".jpg", ".jpeg", ".png")) \
            if p.is_dir() else [p]
    samples = []
    for f in files:
        m = re.match(r"^(\d+)_(\d+)\.", f.name)
        pid, idx = (m.group(1), int(m.group(2))) if m else (f.stem, 0)
        samples.append(Sample(f, -1, pid, idx))
    return samples


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = get_args(argv)
    device = get_device()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.runs_dir)
    model, ckpt = load_model(ckpt_path, device)
    T = ckpt.get("temperature", 1.0)
    targs = ckpt["args"]
    img_size = targs["img_size"]
    root = args.root or targs["root"]
    out_dir = next_run_dir(args.out_dir or ckpt_path.parent, "predict")
    tf = build_transforms(ckpt["data_stats"]["mean"], ckpt["data_stats"]["std"], img_size, train=False)

    print(f"checkpoint {ckpt_path}  ->  results in {out_dir}")
    print(f"model {ckpt['model_name']} (epoch {ckpt['epoch']}), temperature T={T:.3f}, device {device}")

    # referral threshold from validation patients (same split as training)
    _, val = patient_split(index_split(root, "train"), targs["val_frac"], targs["seed"])
    val_pts = predict_split(model, val, tf, args, device, T)
    tau, val_cov, val_acc = choose_threshold(val_pts, args.target_acc)
    print(f"referral threshold tau = {tau:.3f}  (validation: {val_cov:.0%} of patients "
          f"auto-decided at {val_acc:.1%} accuracy, target {args.target_acc:.0%})")

    # ---------- custom images: no labels ----------
    if args.images:
        samples = collect_images(args.images)
        pts = predict_split(model, samples, tf, args, device, T)
        print(f"\n{'case':>12s} {'slices':>6s} {'P(AD)':>6s} {'conf':>5s}  decision")
        for r in pts:
            dec = LABELS[r["pred"]] if r["confidence"] >= tau else "REFER to human"
            print(f"{r['pid']:>12s} {r['n_slices']:6d} {r['p_ad']:6.3f} {r['confidence']:5.2f}  {dec}")
        groups = [("Predictions", pts[i:i + 3]) for i in range(0, min(len(pts), 9), 3)]
        plot_examples(groups, tau, img_size, out_dir / "custom_examples.png", has_labels=False)
        print(f"saved {out_dir / 'custom_examples.png'}")
        return

    # ---------- held-out test set ----------
    test = index_split(root, "test")
    pts = predict_split(model, test, tf, args, device, T)
    n = len(pts)
    correct = [r["pred"] == r["label"] for r in pts]
    auto = [r for r in pts if r["confidence"] >= tau]
    referred = [r for r in pts if r["confidence"] < tau]
    auto_acc = sum(r["pred"] == r["label"] for r in auto) / max(1, len(auto))
    errors_caught = sum(r["pred"] != r["label"] for r in referred)

    print(f"\nTEST: {n} patients")
    print(f"  accuracy, all patients:          {sum(correct) / n:.3f}")
    print(f"  auto-decided:                    {len(auto)} ({len(auto) / n:.0%}), accuracy {auto_acc:.3f}")
    print(f"  referred to human:               {len(referred)} ({len(referred) / n:.0%})")
    print(f"  errors caught by referral:       {errors_caught} of {n - sum(correct)}")
    for cls in (1, 0):
        rs = [r for r in auto if r["label"] == cls]
        if rs:
            print(f"  auto-decided {LABELS[cls]} recall:          "
                  f"{sum(r['pred'] == cls for r in rs) / len(rs):.3f}  (n={len(rs)})")

    print(f"\n{'patient':>10s} {'true':>4s} {'P(AD)':>6s} {'conf':>5s} {'slice sd':>8s}  decision")
    shown = sorted(pts, key=lambda r: r["confidence"])[:args.show]
    for r in shown:
        dec = LABELS[r["pred"]] if r["confidence"] >= tau else "REFER"
        mark = "" if dec == "REFER" else ("  ok" if r["pred"] == r["label"] else "  WRONG")
        print(f"{r['pid']:>10s} {LABELS[r['label']]:>4s} {r['p_ad']:6.3f} {r['confidence']:5.2f} "
              f"{r['slice_std']:8.3f}  {dec}{mark}")
    print(f"(the {len(shown)} least confident patients; full list in predictions.csv)")

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "label", "p_ad", "confidence", "predicted", "decision",
                    "correct", "slice_std", "n_slices"])
        for r in pts:
            dec = LABELS[r["pred"]] if r["confidence"] >= tau else "REFER"
            w.writerow([r["pid"], LABELS[r["label"]], f"{r['p_ad']:.4f}", f"{r['confidence']:.4f}",
                        LABELS[r["pred"]], dec, int(r["pred"] == r["label"]),
                        f"{r['slice_std']:.4f}", r["n_slices"]])

    plot_examples(pick_examples(pts, tau), tau, img_size, out_dir / "examples.png")
    plot_confidence(pts, tau, out_dir / "confidence.png")
    plot_filters(model, out_dir / "global_filters.png")
    print(f"\nsaved examples.png, confidence.png, global_filters.png, predictions.csv to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
