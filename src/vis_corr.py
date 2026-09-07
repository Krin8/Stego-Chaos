"""
Accurate Visualization of Self & KNN Correspondence
=====================================================
Shows exactly what happens during STEGO/EAGLE training:

Row 1: Inputs – anchor image + self-pair (same patient, different slice)
                              + KNN-pair (different patient, same modality)
Row 2: Self-correspondence – feature correlation between anchor & self-pair
Row 3: KNN correspondence  – feature correlation between anchor & KNN-pair
Row 4: Contrastive loss mechanism – pos_intra / pos_inter / neg_inter shifts

Uses the actual BiomedCLIP backbone to extract 768-d features, then
`tensor_correlation(norm(f1), norm(f2))` exactly as ContrastiveCorrelationLoss does.
"""

import sys
import os

# Force UTF-8 on Windows console to avoid cp1252 encoding errors
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure the src directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms as T

# ── Local imports (from the project) ─────────────────────────────────
from utils import prep_for_plot
from data import ContrastiveSegDataset, create_chaos_colormap


# ── Re-implement the exact functions from modules.py ─────────────────
def norm(t):
    """Exactly as in modules.py"""
    return F.normalize(t, dim=1, eps=1e-10)


def tensor_correlation(a, b):
    """Exactly as in modules.py — einsum 'nchw,ncij->nhwij'"""
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    """Exactly as in modules.py"""
    return F.grid_sample(t, coords.permute(0, 2, 1, 3),
                         padding_mode='reflection', align_corners=True)


# ── Build the BiomedCLIP featurizer just like train_segmentation.py ──
def build_featurizer(cfg):
    """Build the same featurizer used during training."""
    from modules import BiomedCLIPFeaturizer
    dim = cfg.dim
    net = BiomedCLIPFeaturizer(dim, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
    return net, device


# ── Minimal config object so we don't need Hydra ────────────────────
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_cfg():
    """Mirror the config from train_config.yaml."""
    return SimpleNamespace(
        pytorch_data_dir="/Users/navneetbavineni/STEGO",
        dataset_name="chaos",
        chaos_modality="CT",
        chaos_n_classes=2,
        use_preprocessed_data=False,
        res=320,
        loader_crop_type="center",
        # Model params
        arch="biomedclip",
        model_type="vit_base",
        projection_type="nonlinear",
        dim=70,
        dropout=False,
        continuous=True,
        dino_patch_size=16,
        dino_feat_type="feat",
        pretrained_weights=None,
        use_text_prompts=False,
        shallow_layer_n=7,
        # Correspondence params (from config)
        pointwise=True,
        feature_samples=11,   # smaller for visualization clarity
        pos_intra_shift=0.18,
        pos_inter_shift=0.12,
        neg_inter_shift=0.46,
        use_salience=False,
        zero_clamp=True,
        stabalize=False,
        # Data params
        num_neighbors=7,
        crop_type="five",
        crop_ratio=0.75,
        batch_size=1,
        num_workers=0,
    )


def get_transform(res, is_label, crop_type):
    """Same as utils.get_transform."""
    from utils import normalize

    class ToTargetTensorLocal:
        def __call__(self, pic):
            return torch.as_tensor(np.array(pic), dtype=torch.int64).unsqueeze(0)

    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        cropper = T.CenterCrop(res)

    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          ToTargetTensorLocal()])
    else:
        normalize_t = T.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          T.ToTensor(),
                          normalize_t])


def find_sample_indices(dataset):
    """
    Find three indices that demonstrate the correspondence pairs:
      1) anchor_idx         – an arbitrary slice with visible organ
      2) self_idx           – same patient, different slice
      3) knn_idx            – different patient, same modality
    Also return a negative (random permutation) index.
    """
    inner = dataset.dataset  # the CHAOS dataset

    # Pick a slice roughly in the middle of a patient so organ is visible
    patients = {}
    for idx, s in enumerate(inner.samples):
        pid = s[3]
        if pid not in patients:
            patients[pid] = []
        patients[pid].append(idx)

    # Pick two patients with enough slices
    good_patients = {p: idxs for p, idxs in patients.items() if len(idxs) >= 5}
    pids = sorted(good_patients.keys())
    if len(pids) < 2:
        raise RuntimeError("Need at least 2 patients with ≥5 slices each")

    pid_a, pid_b = pids[0], pids[1]

    # Anchor = middle slice of patient A
    slices_a = good_patients[pid_a]
    anchor_idx = slices_a[len(slices_a) // 2]

    # Self = different slice, same patient (nearby slice)
    self_idx = slices_a[len(slices_a) // 2 + 2]  # 2 slices away

    # KNN = middle slice of patient B (different patient, same modality)
    slices_b = good_patients[pid_b]
    knn_idx = slices_b[len(slices_b) // 2]

    # Negative = a distant patient
    neg_pid = pids[-1] if len(pids) > 2 else pids[1]
    slices_neg = good_patients.get(neg_pid, good_patients[pid_b])
    neg_idx = slices_neg[0]  # first slice — maximally different

    return anchor_idx, self_idx, knn_idx, neg_idx, pid_a, pid_b, neg_pid


def extract_features(net, img_tensor, device):
    """Run image through BiomedCLIP and return (backbone_feats, projected_code)."""
    img = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(img)
        if len(out) == 3:
            feats, code, _ = out
        else:
            feats, code = out
    return feats, code


def compute_correlation_map(feats_a, feats_b, query_h, query_w):
    """
    Compute the feature-space correlation between a single query position
    in feats_a and ALL positions in feats_b.

    Returns a 2D similarity map of shape (H_b, W_b).
    """
    # feats_a: (1, C, H, W),  feats_b: (1, C, H, W)
    fa = norm(feats_a)
    fb = norm(feats_b)

    # Query vector at (query_h, query_w) from feats_a
    query = fa[:, :, query_h, query_w]  # (1, C)

    # Dot product with every position in feats_b
    sim = torch.einsum("bc,bchw->bhw", query, fb)  # (1, H, W)
    return sim.squeeze(0).cpu().numpy()


def compute_full_correlation(feats_a, feats_b):
    """
    Compute the full tensor correlation as used in ContrastiveCorrelationLoss.
    Returns shape (H_a, W_a, H_b, W_b) — correlation from every position in A
    to every position in B.
    """
    corr = tensor_correlation(norm(feats_a), norm(feats_b))  # (1, H, W, H, W)
    return corr.squeeze(0).cpu()


def main():
    print("=" * 70)
    print("ACCURATE CORRESPONDENCE VISUALIZATION")
    print("Using actual BiomedCLIP features, not pixel-space similarity")
    print("=" * 70)

    cfg = make_cfg()

    # ── 1. Build dataset (same as training) ──────────────────────────
    print("\n[1/5] Loading CHAOS dataset...")
    transform = get_transform(cfg.res, False, cfg.loader_crop_type)
    target_transform = get_transform(cfg.res, True, cfg.loader_crop_type)

    dataset = ContrastiveSegDataset(
        pytorch_data_dir=cfg.pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=transform,
        target_transform=target_transform,
        cfg=cfg,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True,
    )
    print(f"   Dataset has {len(dataset)} samples, {dataset.n_classes} classes")

    # ── 2. Pick representative indices ───────────────────────────────
    print("\n[2/5] Selecting anchor, self-pair, KNN-pair, and negative...")
    anchor_idx, self_idx, knn_idx, neg_idx, pid_a, pid_b, neg_pid = \
        find_sample_indices(dataset)

    inner = dataset.dataset
    print(f"   Anchor : idx={anchor_idx}  patient={pid_a}  "
          f"file={Path(inner.samples[anchor_idx][0]).name}")
    print(f"   Self   : idx={self_idx}  patient={pid_a}  "
          f"file={Path(inner.samples[self_idx][0]).name}")
    print(f"   KNN    : idx={knn_idx}  patient={pid_b}  "
          f"file={Path(inner.samples[knn_idx][0]).name}")
    print(f"   Neg    : idx={neg_idx}  patient={neg_pid}  "
          f"file={Path(inner.samples[neg_idx][0]).name}")

    # Load raw data (img, label, mask)
    anchor_pack = inner[anchor_idx]
    self_pack   = inner[self_idx]
    knn_pack    = inner[knn_idx]
    neg_pack    = inner[neg_idx]

    # ── 3. Build featurizer ──────────────────────────────────────────
    print("\n[3/5] Loading BiomedCLIP backbone...")
    net, device = build_featurizer(cfg)
    print(f"   Running on: {device}")

    # ── 4. Extract features ──────────────────────────────────────────
    print("\n[4/5] Extracting features...")
    feats_anchor, code_anchor = extract_features(net, anchor_pack[0], device)
    feats_self,   code_self   = extract_features(net, self_pack[0], device)
    feats_knn,    code_knn    = extract_features(net, knn_pack[0], device)
    feats_neg,    code_neg    = extract_features(net, neg_pack[0], device)

    feat_h, feat_w = feats_anchor.shape[2], feats_anchor.shape[3]
    print(f"   Feature map size: {feat_h}×{feat_w}  "
          f"(from {cfg.res}×{cfg.res} input, patch_size=16)")

    # Pick query position: center of the feature map
    query_h, query_w = feat_h // 2, feat_w // 2
    # Map back to pixel coordinates for visualization
    patch_size = 16
    query_pixel_y = query_h * patch_size + patch_size // 2
    query_pixel_x = query_w * patch_size + patch_size // 2

    # ── 5. Compute correlations ──────────────────────────────────────
    print("\n[5/5] Computing feature correlations...")

    # Self-correspondence: anchor ↔ self (same patient, different slice)
    self_corr_map = compute_correlation_map(
        feats_anchor, feats_self, query_h, query_w)

    # KNN correspondence: anchor ↔ knn (different patient, same modality)
    knn_corr_map = compute_correlation_map(
        feats_anchor, feats_knn, query_h, query_w)

    # Negative: anchor ↔ neg (random permutation — should be low similarity)
    neg_corr_map = compute_correlation_map(
        feats_anchor, feats_neg, query_h, query_w)

    # Also compute anchor self-self for reference
    self_self_map = compute_correlation_map(
        feats_anchor, feats_anchor, query_h, query_w)

    # Code-space correlations (what the loss actually optimizes)
    code_self_corr = compute_correlation_map(
        code_anchor, code_self, query_h, query_w)
    code_knn_corr = compute_correlation_map(
        code_anchor, code_knn, query_h, query_w)
    code_neg_corr = compute_correlation_map(
        code_anchor, code_neg, query_h, query_w)

    # ══════════════════════════════════════════════════════════════════
    #  VISUALIZATION
    # ══════════════════════════════════════════════════════════════════
    print("\nGenerating visualization...")

    label_cmap = create_chaos_colormap()

    def show_img(ax, img_tensor, title, query_marker=False):
        """Display a normalized image tensor."""
        plot = prep_for_plot(img_tensor)
        ax.imshow(plot)
        ax.set_title(title, fontsize=9, fontweight='bold')
        if query_marker:
            ax.plot(query_pixel_x, query_pixel_y, 'r+', markersize=15, markeredgewidth=2)
            rect = plt.Rectangle(
                (query_pixel_x - patch_size // 2, query_pixel_y - patch_size // 2),
                patch_size, patch_size, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')

    def show_label(ax, label_tensor, title):
        """Display a label map using the CHAOS colormap."""
        lbl = label_tensor.numpy()
        lbl_vis = np.zeros((*lbl.shape, 3), dtype=np.uint8)
        for c in range(len(label_cmap)):
            lbl_vis[lbl == c] = label_cmap[c]
        ax.imshow(lbl_vis)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')

    def show_corr(ax, corr_map, title, vmin=None, vmax=None):
        """Display a correlation map."""
        im = ax.imshow(corr_map, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.axis('off')
        return im

    def find_best_match(corr_map):
        """Find the position of highest correlation."""
        idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
        return idx  # (h, w) in feature-map space

    # Global vmin/vmax for consistent color scales
    all_corr = np.concatenate([self_corr_map.ravel(), knn_corr_map.ravel(),
                                neg_corr_map.ravel()])
    global_vmin = np.percentile(all_corr, 2)
    global_vmax = np.percentile(all_corr, 98)

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle("Accurate Correspondence Visualization\n"
                 "(Using BiomedCLIP 768-d Features, Exactly as in Training)",
                 fontsize=14, fontweight='bold', y=0.98)

    # ── ROW 1: Input images and labels ───────────────────────────────
    gs = fig.add_gridspec(5, 6, hspace=0.35, wspace=0.25,
                          top=0.93, bottom=0.03, left=0.03, right=0.97)

    ax = fig.add_subplot(gs[0, 0])
    show_img(ax, anchor_pack[0], f"Anchor Image\n(Patient {pid_a})", query_marker=True)

    ax = fig.add_subplot(gs[0, 1])
    show_label(ax, anchor_pack[1], f"Anchor Label\n(Patient {pid_a})")

    ax = fig.add_subplot(gs[0, 2])
    show_img(ax, self_pack[0], f"Self-Pair Image\n(Patient {pid_a}, diff slice)")

    ax = fig.add_subplot(gs[0, 3])
    show_label(ax, self_pack[1], f"Self-Pair Label\n(Patient {pid_a})")

    ax = fig.add_subplot(gs[0, 4])
    show_img(ax, knn_pack[0], f"KNN-Pair Image\n(Patient {pid_b})")

    ax = fig.add_subplot(gs[0, 5])
    show_label(ax, knn_pack[1], f"KNN-Pair Label\n(Patient {pid_b})")

    # ── ROW 2: Self-Correspondence (same patient, diff slice) ────────
    # Title for the row
    ax = fig.add_subplot(gs[1, 0])
    show_img(ax, anchor_pack[0], "Anchor\n(query patch marked)", query_marker=True)

    ax = fig.add_subplot(gs[1, 1])
    show_img(ax, self_pack[0], f"Self-Pair\n(same patient {pid_a})")
    # Mark the best-match position
    best_h, best_w = find_best_match(self_corr_map)
    best_py = best_h * patch_size + patch_size // 2
    best_px = best_w * patch_size + patch_size // 2
    ax.plot(best_px, best_py, 'g+', markersize=15, markeredgewidth=2)
    rect = plt.Rectangle(
        (best_px - patch_size // 2, best_py - patch_size // 2),
        patch_size, patch_size, linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

    ax = fig.add_subplot(gs[1, 2])
    im = show_corr(ax, self_corr_map,
                   "Self-Corr: Backbone Features\n(anchor query → self-pair)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 3])
    im = show_corr(ax, code_self_corr,
                   "Self-Corr: Projected Code\n(what loss optimizes)",
                   None, None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 4])
    im = show_corr(ax, self_self_map,
                   "Self-Self: Anchor → Anchor\n(intra-image reference)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Stats
    ax = fig.add_subplot(gs[1, 5])
    ax.axis('off')
    stats_text = (
        f"Self-Correspondence Stats\n"
        f"{'─' * 30}\n"
        f"Pair: same patient, diff slice\n"
        f"Backbone corr mean: {self_corr_map.mean():.3f}\n"
        f"Backbone corr std:  {self_corr_map.std():.3f}\n"
        f"Backbone corr max:  {self_corr_map.max():.3f}\n"
        f"Code corr mean:     {code_self_corr.mean():.3f}\n"
        f"Best match at:      ({best_h}, {best_w})\n"
        f"\nUsed in training as:\n"
        f"  pos_intra_loss (shift={cfg.pos_intra_shift})\n"
        f"  pos_inter_loss (shift={cfg.pos_inter_shift})\n"
        f"  Weight: intra={0.5}, inter={0.15}"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── ROW 3: KNN Correspondence (different patient) ────────────────
    ax = fig.add_subplot(gs[2, 0])
    show_img(ax, anchor_pack[0], "Anchor\n(query patch marked)", query_marker=True)

    ax = fig.add_subplot(gs[2, 1])
    show_img(ax, knn_pack[0], f"KNN-Pair\n(patient {pid_b}, same modality)")
    best_h_k, best_w_k = find_best_match(knn_corr_map)
    best_py_k = best_h_k * patch_size + patch_size // 2
    best_px_k = best_w_k * patch_size + patch_size // 2
    ax.plot(best_px_k, best_py_k, 'g+', markersize=15, markeredgewidth=2)
    rect = plt.Rectangle(
        (best_px_k - patch_size // 2, best_py_k - patch_size // 2),
        patch_size, patch_size, linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

    ax = fig.add_subplot(gs[2, 2])
    im = show_corr(ax, knn_corr_map,
                   "KNN-Corr: Backbone Features\n(anchor query → KNN-pair)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[2, 3])
    im = show_corr(ax, code_knn_corr,
                   "KNN-Corr: Projected Code\n(what loss optimizes)",
                   None, None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[2, 4])
    im = show_corr(ax, neg_corr_map,
                   f"Negative: Anchor → Patient {neg_pid}\n(should be low)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Stats
    ax = fig.add_subplot(gs[2, 5])
    ax.axis('off')
    stats_text = (
        f"KNN-Correspondence Stats\n"
        f"{'─' * 30}\n"
        f"Pair: diff patient, same modality\n"
        f"Backbone corr mean: {knn_corr_map.mean():.3f}\n"
        f"Backbone corr std:  {knn_corr_map.std():.3f}\n"
        f"Backbone corr max:  {knn_corr_map.max():.3f}\n"
        f"Code corr mean:     {code_knn_corr.mean():.3f}\n"
        f"Best match at:      ({best_h_k}, {best_w_k})\n"
        f"\nNegative corr mean: {neg_corr_map.mean():.3f}\n"
        f"  (neg_inter shift={cfg.neg_inter_shift})\n"
        f"  Weight: neg={0.1}"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # ── ROW 4: Contrastive Loss Mechanism ────────────────────────────
    # Show how pos_intra, pos_inter, neg_inter losses work
    # by sampling random coordinate pairs (just like the actual loss)

    coord_shape = [1, cfg.feature_samples, cfg.feature_samples, 2]
    coords1 = torch.rand(coord_shape, device=device) * 2 - 1
    coords2 = torch.rand(coord_shape, device=device) * 2 - 1

    # Sample features at random coordinates
    sampled_feats_a = sample(feats_anchor, coords1)
    sampled_code_a = sample(code_anchor, coords1)
    sampled_feats_s = sample(feats_self, coords2)
    sampled_code_s = sample(code_self, coords2)
    sampled_feats_k = sample(feats_knn, coords2)
    sampled_code_k = sample(code_knn, coords2)

    # Feature-space correlations (frozen backbone — this is fd)
    fd_self = tensor_correlation(norm(sampled_feats_a), norm(sampled_feats_s))
    fd_knn  = tensor_correlation(norm(sampled_feats_a), norm(sampled_feats_k))

    # Code-space correlations (learned projections — this is cd)
    cd_self = tensor_correlation(norm(sampled_code_a), norm(sampled_code_s))
    cd_knn  = tensor_correlation(norm(sampled_code_a), norm(sampled_code_k))

    # Reshape for visualization
    n = cfg.feature_samples
    fd_self_2d = fd_self.squeeze(0).reshape(n * n, n * n).cpu().numpy()
    fd_knn_2d  = fd_knn.squeeze(0).reshape(n * n, n * n).cpu().numpy()
    cd_self_2d = cd_self.squeeze(0).reshape(n * n, n * n).cpu().numpy()
    cd_knn_2d  = cd_knn.squeeze(0).reshape(n * n, n * n).cpu().numpy()

    ax = fig.add_subplot(gs[3, 0])
    im = ax.imshow(fd_self_2d, cmap='RdBu_r', aspect='auto')
    ax.set_title("fd (backbone corr)\nAnchor ↔ Self-pair", fontsize=8, fontweight='bold')
    ax.set_xlabel(f"{n}×{n} sampled positions", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[3, 1])
    im = ax.imshow(cd_self_2d, cmap='RdBu_r', aspect='auto')
    ax.set_title("cd (code corr)\nAnchor ↔ Self-pair", fontsize=8, fontweight='bold')
    ax.set_xlabel(f"{n}×{n} sampled positions", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[3, 2])
    im = ax.imshow(fd_knn_2d, cmap='RdBu_r', aspect='auto')
    ax.set_title("fd (backbone corr)\nAnchor ↔ KNN-pair", fontsize=8, fontweight='bold')
    ax.set_xlabel(f"{n}×{n} sampled positions", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[3, 3])
    im = ax.imshow(cd_knn_2d, cmap='RdBu_r', aspect='auto')
    ax.set_title("cd (code corr)\nAnchor ↔ KNN-pair", fontsize=8, fontweight='bold')
    ax.set_xlabel(f"{n}×{n} sampled positions", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Loss component visualization
    ax = fig.add_subplot(gs[3, 4:])
    ax.axis('off')

    # Compute actual loss values
    pos_intra_loss_val = -(cd_self.clamp(0) * (fd_self - cfg.pos_intra_shift)).mean().item()
    pos_inter_loss_val = -(cd_self.clamp(0) * (fd_self - cfg.pos_inter_shift)).mean().item()

    # For neg, use a permuted sample
    perm = torch.randperm(sampled_feats_a.shape[0], device=device)
    fd_neg = tensor_correlation(norm(sampled_feats_a), norm(sampled_feats_k[perm]))
    cd_neg = tensor_correlation(norm(sampled_code_a), norm(sampled_code_k[perm]))
    neg_inter_loss_val = -(cd_neg.clamp(0) * (fd_neg - cfg.neg_inter_shift)).mean().item()

    loss_text = (
        f"Contrastive Correlation Loss Mechanism\n"
        f"{'═' * 45}\n\n"
        f"loss = -cd.clamp(0) × (fd - shift)\n\n"
        f"Where:\n"
        f"  fd = backbone feature correlation (frozen)\n"
        f"  cd = projected code correlation (learned)\n"
        f"  shift = threshold separating pos/neg signal\n\n"
        f"{'─' * 45}\n"
        f"pos_intra (self-pair, shift={cfg.pos_intra_shift}):\n"
        f"  loss = {pos_intra_loss_val:+.4f}  ×  weight={0.5}\n"
        f"  → encourages cd>0 where fd>{cfg.pos_intra_shift}\n\n"
        f"pos_inter (self-pair, shift={cfg.pos_inter_shift}):\n"
        f"  loss = {pos_inter_loss_val:+.4f}  ×  weight={0.15}\n"
        f"  → wider positive region (lower threshold)\n\n"
        f"neg_inter (random pair, shift={cfg.neg_inter_shift}):\n"
        f"  loss = {neg_inter_loss_val:+.4f}  ×  weight={0.1}\n"
        f"  → pushes apart unrelated images\n"
        f"{'─' * 45}\n"
        f"Both pos_intra and pos_inter use SAME pair!\n"
        f"Only the shift differs (not self vs KNN)."
    )
    ax.text(0.05, 0.95, loss_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

    # ── ROW 5: Summary diagram ───────────────────────────────────────
    ax = fig.add_subplot(gs[4, :])
    ax.axis('off')

    summary = (
        "TRAINING DATA FLOW SUMMARY\n"
        "═══════════════════════════════════════════════════════════════════════════════════════════════════════════════\n"
        "\n"
        "  ┌─────────────────┐     Self-pair: same patient, diff slice      ┌─────────────────┐\n"
        "  │  Anchor Image   │ ──────────────────────────────────────────── │  Self Image     │\n"
        "  │  (Patient A,    │     Used for pos_intra & pos_inter losses    │  (Patient A,    │\n"
        "  │   slice k)      │                                              │   slice k±Δ)    │\n"
        "  └────────┬────────┘                                              └─────────────────┘\n"
        "           │\n"
        "           │  KNN-pair: same modality, diff patient\n"
        "           │  (This is what img_pos actually is in the dataloader)\n"
        "           ▼\n"
        "  ┌─────────────────┐     Negative: random permutation within batch ┌─────────────────┐\n"
        "  │  KNN Image      │ ────────────────────────────────────────────  │  Neg Image      │\n"
        "  │  (Patient B,    │     Used for neg_inter loss                   │  (Patient C,    │\n"
        "  │   same modality)│                                               │   any slice)    │\n"
        "  └─────────────────┘                                               └─────────────────┘\n"
    )
    ax.text(0.02, 0.95, summary, transform=ax.transAxes,
            fontsize=7.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#fffff0', alpha=0.9))

    # ── Save ─────────────────────────────────────────────────────────
    output_dir = Path("/Users/navneetbavineni/STEGO/correspondence_visualization")
    output_dir.mkdir(exist_ok=True)

    save_path = output_dir / "accurate_correspondence.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Saved to: {save_path}")

    plt.close(fig)

    # ── Also save a compact version with just the key comparisons ────
    fig2, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig2.suptitle("Correspondence Comparison: Self vs KNN vs Negative\n"
                  "(BiomedCLIP backbone features, query = red cross)",
                  fontsize=13, fontweight='bold')

    # Row 1: Self-correspondence
    show_img(axes[0, 0], anchor_pack[0], f"Anchor (Patient {pid_a})", query_marker=True)
    show_img(axes[0, 1], self_pack[0], f"Self: Same Patient {pid_a}\n(different slice)")
    im = show_corr(axes[0, 2], self_corr_map, "Feature Correlation\n(Anchor → Self)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    show_label(axes[0, 3], self_pack[1], f"Self-Pair Ground Truth")

    # Row 2: KNN-correspondence
    show_img(axes[1, 0], anchor_pack[0], f"Anchor (Patient {pid_a})", query_marker=True)
    show_img(axes[1, 1], knn_pack[0], f"KNN: Patient {pid_b}\n(same modality)")
    im = show_corr(axes[1, 2], knn_corr_map, "Feature Correlation\n(Anchor → KNN)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    show_label(axes[1, 3], knn_pack[1], f"KNN-Pair Ground Truth")

    axes[0, 0].set_ylabel("SELF\n(same patient,\ndiff slice)", fontsize=11,
                          fontweight='bold', color='blue')
    axes[1, 0].set_ylabel("KNN\n(diff patient,\nsame modality)", fontsize=11,
                          fontweight='bold', color='green')

    plt.tight_layout()
    save_path2 = output_dir / "correspondence_compact.png"
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved to: {save_path2}")

    plt.close(fig2)

    # ── Print explanation ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HOW CORRESPONDENCE WORKS IN YOUR MODEL")
    print("=" * 70)
    print("""
DATA PAIRING (in ContrastiveSegDataset.__getitem__):
  - img       = anchor slice (e.g., Patient A, slice k)
  - img_pos   = KNN positive (same modality, DIFFERENT patient)
  - img_aug   = augmented version of a SELF slice (same patient, diff slice)

CONTRASTIVE CORRELATION LOSS (in ContrastiveCorrelationLoss.forward):
  Uses (img, img_pos) pair for BOTH pos_intra and pos_inter:
  
  1. pos_intra_loss: -cd * (fd - 0.18)  <-- strict threshold
     -> Only positions with backbone similarity > 0.18 get positive signal
     -> Focuses on EXACT anatomical matches
     
  2. pos_inter_loss: -cd * (fd - 0.12)  <-- looser threshold
     -> Positions with backbone similarity > 0.12 get positive signal  
     -> Captures broader structural correspondence
     
  3. neg_inter_loss: -cd * (fd - 0.46)  <-- high threshold on RANDOM pair
     -> Since random pairs rarely have fd > 0.46, this term
       pushes cd NEGATIVE for unrelated positions
       -> Prevents the code from collapsing to uniform similarity

WHERE:
  fd = tensor_correlation(norm(backbone_features_A), norm(backbone_features_B))
       ^ FROZEN -- comes from BiomedCLIP, provides the "teacher" signal
  cd = tensor_correlation(norm(projected_code_A), norm(projected_code_B))  
       ^ LEARNED -- this is what the model optimizes
""")


if __name__ == "__main__":
    main()