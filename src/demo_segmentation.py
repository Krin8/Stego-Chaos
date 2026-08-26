import os

import numpy as np
import torch
import torch.nn.functional as F
from os.path import join
from modules import *
import hydra
import torch.multiprocessing
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from train_segmentation import LitUnsupervisedSegmenter, prep_for_plot
from tqdm import tqdm
import random
from data import create_chaos_colormap, ContrastiveSegDataset, CHAOS
from scipy import ndimage

try:
    import pydicom
except ImportError as e:
    raise ImportError("pydicom is required to read DICOM inputs; install it with "
                      "`pip install pydicom`") from e

torch.multiprocessing.set_sharing_strategy('file_system')


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clean_segmentation(seg_map, n_classes, min_area=50):
    struct = ndimage.generate_binary_structure(2, 2)
    cleaned = np.zeros_like(seg_map)
    for cls in range(1, n_classes):
        mask = (seg_map == cls).astype(np.uint8)
        if mask.sum() == 0:
            continue
        mask = ndimage.binary_opening(mask, structure=struct, iterations=1)
        mask = ndimage.binary_closing(mask, structure=struct, iterations=1)
        labeled, n_components = ndimage.label(mask)
        for comp_id in range(1, n_components + 1):
            if (labeled == comp_id).sum() < min_area:
                mask[labeled == comp_id] = 0
        cleaned[mask > 0] = cls
    return cleaned


class DicomImageFolder(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.transform = transform
        if not os.path.isdir(root):
            raise FileNotFoundError(f"DICOM image directory not found: {root}")

        self.dicom_paths = []
        self._discover(root)
        if not self.dicom_paths:
            raise RuntimeError(f"No .dcm files found under {root}")
        print(f"[DicomImageFolder] Found {len(self.dicom_paths)} DICOM files in {root}")

    def _discover(self, path):
        for entry in sorted(os.listdir(path)):
            full = join(path, entry)
            if os.path.isdir(full):
                self._discover(full)
            elif entry.lower().endswith(".dcm"):
                self.dicom_paths.append(full)

    @staticmethod
    def _load_dicom_as_pil(dcm_path):
        dcm = pydicom.dcmread(dcm_path)
        arr = dcm.pixel_array.astype(np.float32)
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            arr = arr * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
        is_ct = hasattr(dcm, 'Modality') and dcm.Modality == 'CT'
        if is_ct:
            wc, ww = 50, 400
            lo, hi = wc - ww / 2, wc + ww / 2
        else:
            lo, hi = np.percentile(arr, (1, 99))
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        img = ImageOps.equalize(img)
        return img.convert("RGB")

    def __getitem__(self, index):
        dcm_path = self.dicom_paths[index]
        
        # Determine modality before pixel reading
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        is_ct = hasattr(dcm, 'Modality') and dcm.Modality == 'CT'
        
        if is_ct:
            image, _ = CHAOS._preprocess_ct_slice(dcm_path, None, self.transform, None, "CT")
        else:
            image = self._load_dicom_as_pil(dcm_path)
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            
        name = os.path.relpath(dcm_path, self.root).replace(os.sep, "_")
        return image, name

    def __len__(self):
        return len(self.dicom_paths)


def build_assignment_lut(raw_assignments, n_clusters):
    if (isinstance(raw_assignments, tuple)
            and len(raw_assignments) == 2
            and isinstance(raw_assignments[0], (np.ndarray, list, torch.Tensor))
            and isinstance(raw_assignments[1], (np.ndarray, list, torch.Tensor))):
        row_ind = np.asarray(raw_assignments[0], dtype=np.int64)
        col_ind = np.asarray(raw_assignments[1], dtype=np.int64)
        lut = np.zeros(n_clusters, dtype=np.int64)
        lut[row_ind] = col_ind
        return lut

    if isinstance(raw_assignments, torch.Tensor):
        raw_assignments = raw_assignments.cpu().numpy()

    if isinstance(raw_assignments, np.ndarray):
        if raw_assignments.ndim == 1 and len(raw_assignments) == n_clusters:
            return raw_assignments.astype(np.int64)
        if raw_assignments.ndim == 2:
            lut = np.zeros(n_clusters, dtype=np.int64)
            for row in raw_assignments:
                lut[int(row[0])] = int(row[1])
            return lut

    if isinstance(raw_assignments, (list, tuple)):
        if len(raw_assignments) == n_clusters and not isinstance(raw_assignments[0], (list, tuple)):
            return np.array(raw_assignments, dtype=np.int64)
        lut = np.zeros(n_clusters, dtype=np.int64)
        for pair in raw_assignments:
            lut[int(pair[0])] = int(pair[1])
        return lut

    raise ValueError(f"Unrecognised assignments format: {type(raw_assignments)}")


def get_cluster_assignments(model, cfg, device):
    """Run validation set through model to compute cluster-to-class LUT."""
    print("Computing cluster assignments from validation set...")

    def make_dataset(image_set):
        return ContrastiveSegDataset(
            pytorch_data_dir=cfg.pytorch_data_dir,
            dataset_name=model.cfg.dataset_name,
            crop_type=None,
            image_set=image_set,
            transform=get_transform(model.cfg.res, False, None),
            target_transform=get_transform(model.cfg.res, True, None),
            mask=True,
            cfg=model.cfg,
        )

    val_dataset = make_dataset("val")
    print(f"  Val dataset size: {len(val_dataset)} samples")

    if len(val_dataset) == 0:
        print("  WARNING: val split empty, trying 'train' split for assignments...")
        val_dataset = make_dataset("train")
        print(f"  Train dataset size: {len(val_dataset)} samples")

    if len(val_dataset) == 0:
        raise RuntimeError(
            "Both the val and train splits are empty, cluster assignments cannot be "
            f"computed. Check pytorch_data_dir ({cfg.pytorch_data_dir}).")

    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    model.cluster_metrics.reset()
    model.eval().to(device)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing assignments"):
            img   = batch["img"].to(device)
            label = batch["label"].to(device)
            feats, code = model.net(img)
            code = F.interpolate(code, label.shape[-2:],
                                 mode='bilinear', align_corners=False)
            _, cluster_preds = model.cluster_probe(code, None)
            model.cluster_metrics.update(cluster_preds.argmax(1), label)

    model.cluster_metrics.compute()

    raw = None
    for attr in ['assignments', 'assignment', 'cluster_to_class', 'perm', 'hungarian_match']:
        if hasattr(model.cluster_metrics, attr):
            raw = getattr(model.cluster_metrics, attr)
            break

    if raw is None:
        raise RuntimeError(
            "No cluster assignment attribute found on cluster_metrics "
            f"({type(model.cluster_metrics).__name__}); cannot map clusters to classes.")

    n_classes      = model.n_classes
    extra_clusters = getattr(model.cfg, 'extra_clusters', 0)
    n_clusters     = n_classes + extra_clusters
    print(f"  Cluster probe size: {n_clusters} ({n_classes} classes + {extra_clusters} extra)")

    lut = build_assignment_lut(raw, n_clusters)

    print("=== Cluster -> Class Assignment Table ===")
    for cid, cls in enumerate(lut):
        print(f"  cluster {cid:2d}  ->  class {int(cls)}")
    foreground = [(cid, int(cls)) for cid, cls in enumerate(lut) if cls > 0]
    print(f"  Foreground-mapped clusters: {foreground}")
    print("=========================================\n")

    return lut, n_classes


def apply_cluster_mapping(cluster_pred_np, lut, n_classes):
    safe = np.clip(cluster_pred_np, 0, len(lut) - 1)
    mapped = lut[safe]
    return np.clip(mapped, 0, n_classes - 1)


@hydra.main(config_path="configs", config_name="demo_config.yaml")
def my_app(cfg: DictConfig) -> None:
    result_dir = join(cfg.output_root, "predictions", cfg.experiment_name)
    os.makedirs(join(result_dir, "grids"), exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    model = LitUnsupervisedSegmenter.load_from_checkpoint(cfg.model_path, weights_only=False)

    label_cmap = create_chaos_colormap()

    lut, n_classes = get_cluster_assignments(model, cfg, device)
    print(f"Number of semantic classes: {n_classes}")

    dataset = DicomImageFolder(
        root=cfg.image_dir,
        transform=get_transform(cfg.res, False, None),
    )

    loader = DataLoader(
        dataset, cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=flexible_collate,
    )

    model.eval().to(device)
    par_model = model.net
    img_idx = 0

    for img, name in tqdm(loader, desc="Generating grids"):
        with torch.no_grad():
            img = img.to(device)

            feats1, code1 = par_model(img)
            feats2, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2
            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

            linear_preds = torch.argmax(
                torch.log_softmax(model.linear_probe(code), dim=1), dim=1
            ).cpu()

            _, cluster_probs = model.cluster_probe(code, None)
            cluster_preds = cluster_probs.argmax(1).cpu()

            for j in range(img.shape[0]):
                input_img = (prep_for_plot(img[j].cpu()).numpy() * 255).clip(0, 255).astype(np.uint8)

                raw_cluster    = cluster_preds[j].numpy()
                cluster_mapped = apply_cluster_mapping(raw_cluster, lut, n_classes)
                cluster_mapped = clean_segmentation(cluster_mapped, n_classes)
                cluster_img    = label_cmap[cluster_mapped]

                linear_pred = clean_segmentation(linear_preds[j].numpy(), n_classes)
                linear_img  = label_cmap[linear_pred]

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for ax, title, panel in zip(axes,
                                            ["Input", "Cluster", "Linear"],
                                            [input_img, cluster_img, linear_img]):
                    ax.imshow(panel)
                    ax.set_title(title, fontsize=16, fontweight='bold')
                    ax.axis('off')

                plt.tight_layout(pad=0.5)
                safe_name = name[j].replace(",", "_").replace("/", "_")
                save_path = join(result_dir, "grids", f"grid_{img_idx:04d}_{safe_name}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight',
                            facecolor='white', pad_inches=0.1)
                plt.close(fig)
                img_idx += 1

    print(f"\nDone! {img_idx} grid images saved to: {join(result_dir, 'grids')}")


if __name__ == "__main__":
    prep_args()
    my_app()