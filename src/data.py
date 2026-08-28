import os
import random
from os.path import join

import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
import pydicom
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

def bit_get(val, idx):
    return (val >> idx) & 1


def create_pascal_label_colormap():
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)
    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3
    return colormap


def create_chaos_colormap():
    """
    0 = background  (black)
    1 = liver        (red)
    2 = right kidney (dark blue)
    3 = left kidney  (light blue)
    4 = spleen       (yellow)
    """
    return np.array([
        [0,   0,   0],
        [255,  0,   0],
        [0,   0, 255],
        [0, 128, 255],
        [255, 255,  0],
    ], dtype=np.uint8)


# ---------------------------------------------------------------------------
# CHAOS label pixel → class index mappings
#
# CT  Ground PNGs:  0 = background, 255 = liver
# MRI Ground PNGs:  0 = background, 63 = liver, 126 = right kidney,
#                   189 = left kidney, 252 = spleen
# ---------------------------------------------------------------------------

_CT_PIXEL_TO_CLASS  = {0: 0, 255: 1}
_MRI_PIXEL_TO_CLASS = {0: 0, 63: 1, 126: 2, 189: 3, 252: 4}


# ---------------------------------------------------------------------------
# CHAOS Dataset
# ---------------------------------------------------------------------------
#
# Exact on-disk layout (verified from screenshots):
#
#   <pytorch_data_dir>/
#     archive/
#       CHAOS_Train_Sets/
#         Train_Sets/
#           CT/
#             1/
#               DICOM_anon/   *.dcm
#               Ground/       liver_GT_000.png, liver_GT_001.png ...
#             2/ 5/ 6/ 8/ ...
#           MR/
#             <pid>/
#               T1DUAL/
#                 DICOM_anon/
#                   InPhase/  *.dcm
#                 Ground/     liver_GT_000.png ...
#               T2SPIR/
#                 DICOM_anon/ *.dcm
#                 Ground/     liver_GT_000.png ...
# ---------------------------------------------------------------------------

class CHAOS(Dataset):
    """
    Parameters
    ----------
    root             : str  – pytorch_data_dir (contains 'archive/')
    modality         : str  – 'CT' | 'T1DUAL' | 'T2SPIR' | 'all'
    image_set        : str  – 'train' | 'val' | 'all'
    transform        : PIL Image -> tensor [3, H, W]
    target_transform : PIL Image -> tensor [1, H, W]  (nearest, no interp)
    n_classes        : int – 2 (CT liver only) or 5 (all MRI organs)
    """

    MODALITIES = ("CT", "T1DUAL", "T2SPIR")

    def __init__(self, root, modality, image_set, transform, target_transform,
                 n_classes=5, use_preprocessed_data=False):
        super().__init__()
        assert modality in (*self.MODALITIES, "all"), \
            f"modality must be one of {self.MODALITIES + ('all',)}"
        assert image_set in ("train", "val", "all")

        self.modality         = modality
        self.image_set        = image_set
        self.transform        = transform
        self.target_transform = target_transform
        self.n_classes        = n_classes
        self.root             = root
        self.use_preprocessed_data = use_preprocessed_data

        # Root of the actual CHAOS train data (verified path)
        self.train_root = join(root, "archive", "CHAOS_Train_Sets", "Train_Sets")

        # Each entry: (dcm_path, mask_path_or_None, modality_tag, patient_id)
        self.samples: list = []
        if self.use_preprocessed_data:
            self._collect_preprocessed_samples()
        else:
            self._collect_samples()

            # Patient-level 80/20 split — all slices of a patient stay together
            # so there is no cross-patient leakage between train and val.
            if image_set != "all":
                all_patients = sorted(set(s[3] for s in self.samples))
                n_val = max(1, int(0.2 * len(all_patients)))
                val_patients = set(all_patients[-n_val:])
                if image_set == "val":
                    self.samples = [s for s in self.samples if s[3] in val_patients]
                else:
                    self.samples = [s for s in self.samples if s[3] not in val_patients]

    # ------------------------------------------------------------------
    # Sample collection — exact paths from screenshots
    # ------------------------------------------------------------------

    def _collect_samples(self):
        mods_wanted = self.MODALITIES if self.modality == "all" \
                      else (self.modality,)

        if "CT" in mods_wanted:
            ct_root = join(self.train_root, "CT")
            if os.path.isdir(ct_root):
                for pid in sorted(os.listdir(ct_root),
                                  key=lambda x: int(x) if x.isdigit() else 0):
                    self._add_ct(join(ct_root, pid), pid)

        for seq in ("T1DUAL", "T2SPIR"):
            if seq in mods_wanted:
                mr_root = join(self.train_root, "MR")
                if os.path.isdir(mr_root):
                    for pid in sorted(os.listdir(mr_root),
                                      key=lambda x: int(x) if x.isdigit() else 0):
                        self._add_mri(join(mr_root, pid), pid, seq)

    def _add_ct(self, patient_path: str, patient_id: str):
        """
        CT/<pid>/DICOM_anon/*.dcm  paired with  CT/<pid>/Ground/liver_GT_NNN.png
        Ground/ contains PNGs directly — no nested subfolder.
        """
        dicom_dir  = join(patient_path, "DICOM_anon")
        ground_dir = join(patient_path, "Ground")

        if not os.path.isdir(dicom_dir):
            return

        dcm_files  = sorted(f for f in os.listdir(dicom_dir)
                             if f.lower().endswith(".dcm"))
        mask_files = sorted(f for f in os.listdir(ground_dir)
                             if f.lower().endswith(".png")) \
                     if os.path.isdir(ground_dir) else []

        for i, dcm in enumerate(dcm_files):
            mask_path = join(ground_dir, mask_files[i]) \
                        if i < len(mask_files) else None
            self.samples.append((join(dicom_dir, dcm), mask_path, "CT", patient_id))

    def _add_mri(self, patient_path: str, patient_id: str, seq: str):
        """
        MR/<pid>/T1DUAL/DICOM_anon/InPhase/*.dcm  ↔  MR/<pid>/T1DUAL/Ground/*.png
        MR/<pid>/T2SPIR/DICOM_anon/*.dcm           ↔  MR/<pid>/T2SPIR/Ground/*.png
        Ground/ contains PNGs directly — no nested patient subfolder.
        """
        seq_path = join(patient_path, seq)
        if not os.path.isdir(seq_path):
            return

        dicom_dir = join(seq_path, "DICOM_anon", "InPhase") \
                    if seq == "T1DUAL" else join(seq_path, "DICOM_anon")
        ground_dir = join(seq_path, "Ground")

        if not os.path.isdir(dicom_dir):
            return

        dcm_files  = sorted(f for f in os.listdir(dicom_dir)
                             if f.lower().endswith(".dcm"))
        mask_files = sorted(f for f in os.listdir(ground_dir)
                             if f.lower().endswith(".png")) \
                     if os.path.isdir(ground_dir) else []

        for i, dcm in enumerate(dcm_files):
            mask_path = join(ground_dir, mask_files[i]) \
                        if i < len(mask_files) else None
            self.samples.append((join(dicom_dir, dcm), mask_path, seq, patient_id))

    # ------------------------------------------------------------------
    # DICOM → 8-bit RGB PIL Image
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dicom_as_pil(dcm_path: str) -> Image.Image:
        dcm = pydicom.dcmread(dcm_path)
        arr = dcm.pixel_array.astype(np.float32)
        
        # Rescale to Hounsfield Units (HU) if metadata exists
        intercept = getattr(dcm, 'RescaleIntercept', 0)
        slope = getattr(dcm, 'RescaleSlope', 1)
        arr = arr * slope + intercept

        # Standard Abdominal Window (Level = 40, Width = 400)
        # This highlights the liver, kidneys, and spleen while ignoring bone/air
        W_level = 40
        W_width = 400
        lower_bound = W_level - (W_width / 2)
        upper_bound = W_level + (W_width / 2)
        
        # Clip and normalize based on the fixed window
        arr = np.clip(arr, lower_bound, upper_bound)
        arr = (arr - lower_bound) / (upper_bound - lower_bound)
        
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    @staticmethod
    def _preprocess_ct_slice(dcm_path: str, mask_path: str, transform, target_transform, modality: str) -> tuple:
        import SimpleITK as sitk
        import cv2

        # 1. LOAD (Internal PNG Conversion)
        sitk_img = sitk.ReadImage(dcm_path)
        img_array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        hu_data = img_array[0]  # Shape: (H, W)

        # 2. RESAMPLE (Standardize to 1.0mm)
        original_spacing = sitk_img.GetSpacing()
        new_size_x = int(round(sitk_img.GetSize()[0] * (original_spacing[0] / 1.0)))
        new_size_y = int(round(sitk_img.GetSize()[1] * (original_spacing[1] / 1.0)))
        resampled_img = cv2.resize(hu_data, (new_size_x, new_size_y), interpolation=cv2.INTER_LINEAR)

        has_mask = mask_path is not None and os.path.exists(mask_path)
        if has_mask:
            # Load and map mask first using static method _load_mask
            mask_pil = CHAOS._load_mask(mask_path, modality)
            mask_raw = np.array(mask_pil)
            resampled_mask = cv2.resize(mask_raw, (new_size_x, new_size_y), interpolation=cv2.INTER_NEAREST)
        else:
            resampled_mask = None

        # 3. CROP (Body Mask ROI)
        mask_for_crop = (resampled_img > -500).astype(np.uint8)
        coords = cv2.findNonZero(mask_for_crop)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped_img = resampled_img[y:y+h, x:x+w]
            if has_mask:
                cropped_mask = resampled_mask[y:y+h, x:x+w]
        else:
            cropped_img = resampled_img
            if has_mask:
                cropped_mask = resampled_mask

        # 4. WINDOW (Soft Tissue [-150, 250])
        windowed = np.clip(cropped_img, -150, 250)

        # 5. NORMALIZE (0 to 1)
        normalized = (windowed - (-150)) / (250 - (-150))
        normalized = np.clip(normalized, 0, 1)

        # 6. SPATIAL STANDARDIZATION (Padding)
        h, w = normalized.shape
        max_dim = max(h, w)
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2

        padded_img = np.pad(
            normalized,
            ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w)),
            mode='constant',
            constant_values=0
        )

        if has_mask:
            padded_mask = np.pad(
                cropped_mask,
                ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w)),
                mode='constant',
                constant_values=255
            )
        else:
            padded_mask = None

        # 7. RESIZE to target_res dynamically extracted from transform
        target_res = 224
        from torchvision.transforms import Resize
        if hasattr(transform, "transforms"):
            for t in transform.transforms:
                if isinstance(t, Resize):
                    target_res = t.size
                    if isinstance(target_res, (list, tuple)):
                        target_res = target_res[0]
                    break

        final_img_2d = cv2.resize(padded_img, (target_res, target_res), interpolation=cv2.INTER_AREA)
        if has_mask:
            final_mask_2d = cv2.resize(padded_mask, (target_res, target_res), interpolation=cv2.INTER_NEAREST)
        else:
            final_mask_2d = None

        # Convert image back to PIL (0-255 uint8)
        final_img_8bit = (final_img_2d * 255).astype(np.uint8)
        img_pil = Image.fromarray(final_img_8bit).convert("RGB")

        seed = np.random.randint(2147483647)
        random.seed(seed); torch.manual_seed(seed)
        img_tensor = transform(img_pil)

        if has_mask:
            mask_pil = Image.fromarray(final_mask_2d, mode="L")
            random.seed(seed); torch.manual_seed(seed)
            label_tensor = target_transform(mask_pil).squeeze(0).long()
            label_tensor[label_tensor == 255] = -1
        else:
            label_tensor = torch.full((target_res, target_res), -1, dtype=torch.long)

        return img_tensor, label_tensor

    # ------------------------------------------------------------------
    # Ground PNG → uint8 class-index PIL Image (mode "L")
    # 255 is used as the ignore sentinel in uint8 space;
    # it is remapped to -1 (int64) inside __getitem__.
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mask(mask_path: str, modality: str) -> Image.Image:
        raw = np.array(Image.open(mask_path).convert("L"))   # uint8 grayscale
        lut = _CT_PIXEL_TO_CLASS if modality == "CT" else _MRI_PIXEL_TO_CLASS
        label = np.full(raw.shape, 255, dtype=np.uint8)      # 255 = ignore
        for px, cls in lut.items():
            label[raw == px] = cls
        return Image.fromarray(label, mode="L")

    def _collect_preprocessed_samples(self):
        parent_dir = os.path.dirname(os.path.abspath(self.root))
        preprocessed_root = join(parent_dir, "preprocessed")
        if not os.path.isdir(preprocessed_root):
            preprocessed_root = join(self.root, "preprocessed")
            
        if self.image_set == "all":
            splits = ["train", "val"]
        else:
            splits = [self.image_set]
            
        mods_wanted = self.MODALITIES if self.modality == "all" else (self.modality,)
        
        for split in splits:
            split_dir = join(preprocessed_root, split)
            img_dir = join(split_dir, "images")
            label_dir = join(split_dir, "labels")
            
            if not os.path.isdir(img_dir):
                continue
                
            for fname in sorted(os.listdir(img_dir)):
                if not fname.lower().endswith(".png"):
                    continue
                parts = fname.split("_")
                if len(parts) < 3:
                    continue
                mod = parts[0]
                pid = parts[1]
                
                if mod in mods_wanted:
                    img_path = join(img_dir, fname)
                    mask_path = join(label_dir, fname)
                    if not os.path.exists(mask_path):
                        mask_path = None
                    self.samples.append((img_path, mask_path, mod, pid))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.use_preprocessed_data:
            img_path, mask_path, mod, _ = self.samples[index]
            img = Image.open(img_path).convert("RGB")

            seed = np.random.randint(2147483647)
            random.seed(seed); torch.manual_seed(seed)
            img = self.transform(img)

            if mask_path is not None and os.path.exists(mask_path):
                mask_pil = self._load_mask(mask_path, mod)
                random.seed(seed); torch.manual_seed(seed)
                label = self.target_transform(mask_pil).squeeze(0).long()
                label[label == 255] = -1
            else:
                label = torch.full(img.shape[1:], -1, dtype=torch.long)

            img_sum = img.sum(0)
            valid_mask = (img_sum > (img_sum.min() + 0.01)).float()
            return img, label, valid_mask

        dcm_path, mask_path, mod, _ = self.samples[index]

        if mod == "CT":
            img, label = self._preprocess_ct_slice(dcm_path, mask_path, self.transform, self.target_transform, mod)
        else:
            img = self._load_dicom_as_pil(dcm_path)

            seed = np.random.randint(2147483647)

            random.seed(seed); torch.manual_seed(seed)
            img = self.transform(img)                            # [3, H, W]

            if mask_path is not None and os.path.exists(mask_path):
                mask_pil = self._load_mask(mask_path, mod)
                random.seed(seed); torch.manual_seed(seed)
                label = self.target_transform(mask_pil).squeeze(0).long()  # [H, W]
                label[label == 255] = -1                         # restore ignore sentinel
            else:
                label = torch.full(img.shape[1:], -1, dtype=torch.long)

        img_sum = img.sum(0)
        # Soft-tissue Intensity Window: Exclude air/lungs (dark) and bone (very bright)
        # img_sum ranges from 0.0 to 3.0. Soft tissue usually lies between 0.3 and 2.5
        valid_mask = ((img_sum > 0.3) & (img_sum < 2.5)).float()
        return img, label, valid_mask


# ---------------------------------------------------------------------------
# MaterializedDataset
# ---------------------------------------------------------------------------

class MaterializedDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.materialized = []
        loader = DataLoader(ds, num_workers=4, collate_fn=lambda l: l[0])
        for batch in tqdm(loader):
            self.materialized.append(batch)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        return self.materialized[ind]


# ---------------------------------------------------------------------------
# ContrastiveSegDataset — EAGLE-compatible wrapper
# ---------------------------------------------------------------------------
#
# In your Hydra / train config set:
#   dataset_name    : chaos
#   chaos_modality  : CT | T1DUAL | T2SPIR | all   (default: all)
#   chaos_n_classes : 5                             (default: 5)
#
# pytorch_data_dir must point to the folder that contains "archive/", i.e.:
#   /content/drive/MyDrive/EAGLE/src_EAGLE/pytorch_data_dir
# ---------------------------------------------------------------------------

class ContrastiveSegDataset(Dataset):

    def __init__(
        self,
        pytorch_data_dir,
        dataset_name,
        crop_type,              # ignored — kept for API compatibility
        image_set,
        transform,
        target_transform,
        cfg,
        aug_geometric_transform=None,
        aug_photometric_transform=None,
        mask=False,
        extra_transform=None,
        model_type_override=None,
        num_neighbors=None,
        pos_images=False,
        pos_labels=False,
    ):
        super().__init__()

        # assert dataset_name == "chaos", \
        #     f"This datasets.py only supports dataset_name='chaos', got '{dataset_name}'"

        self.mask                      = mask
        self.extra_transform           = extra_transform
        self.aug_geometric_transform   = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform
        self.pos_images                = pos_images
        self.pos_labels                = pos_labels

        modality  = getattr(cfg, "chaos_modality",  "all")
        n_classes = getattr(cfg, "chaos_n_classes", 5)
        use_preprocessed_data = getattr(cfg, "use_preprocessed_data", False)
        self.n_classes = n_classes

        self.dataset = CHAOS(
            root             = pytorch_data_dir,
            modality         = modality,
            image_set        = image_set,
            transform        = transform,
            target_transform = target_transform,
            n_classes        = n_classes,
            use_preprocessed_data = use_preprocessed_data,
        )

        # Build patient → indices map for positive pair sampling
        self._patient_to_indices = {}
        for idx, sample in enumerate(self.dataset.samples):
            pid = sample[3]  # patient_id
            if pid not in self._patient_to_indices:
                self._patient_to_indices[pid] = []
            self._patient_to_indices[pid].append(idx)
        # Pre-compute per-index lists for O(1) lookup
        self._pos_candidates = []
        for idx, sample in enumerate(self.dataset.samples):
            pid = sample[3]
            siblings = [i for i in self._patient_to_indices[pid] if i != idx]
            if len(siblings) == 0:
                siblings = [idx]  # fallback: use self if only one slice
            self._pos_candidates.append(siblings)

        # Load precomputed KNN neighbors if available
        self.num_neighbors = num_neighbors
        self.nns = None
        if num_neighbors is not None and num_neighbors > 0:
            model_type = getattr(cfg, "model_type", "vit_base")
            nice_name = getattr(cfg, "dir_dataset_name", None) if dataset_name == "directory" else dataset_name
            crop_t = getattr(cfg, "crop_type", "five")
            nns_file = os.path.join(pytorch_data_dir, "nns",
                                    f"nns_{model_type}_{nice_name}_{image_set}_{crop_t}_224.npz")
            if os.path.exists(nns_file):
                self.nns = torch.from_numpy(np.load(nns_file)["nns"])
                print(f"Loaded KNN neighbors from {nns_file}, shape={self.nns.shape}")
            else:
                print(f"KNN file not found: {nns_file}, falling back to patient-based sampling")

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, ind):
        pack = self.dataset[ind]   # (img, label, valid_mask)

        seed = np.random.randint(2147483647)
        self._set_seed(seed)

        # Sample a positive pair using KNN neighbors or patient-based fallback
        if self.pos_images:
            if self.nns is not None:
                # Use precomputed KNN: pick a random neighbor from top-k
                # Skip index 0 (self), pick from indices 1..num_neighbors
                knn_neighbors = self.nns[ind]
                k = min(self.num_neighbors, len(knn_neighbors))
                pos_idx = knn_neighbors[random.randint(1, k - 1)].item()
                pos_idx = min(pos_idx, len(self.dataset) - 1)  # bounds check
            else:
                # Fallback: patient-based sampling
                pos_idx = random.choice(self._pos_candidates[ind])
            pos_pack = self.dataset[pos_idx]
        else:
            pos_pack = pack

        # Spatial coordinate grid — used by some EAGLE loss terms
        coord_entries = torch.meshgrid(
            [torch.linspace(-1, 1, pack[0].shape[1]),
             torch.linspace(-1, 1, pack[0].shape[2])],
            indexing="ij"
        )
        coord = torch.cat([t.unsqueeze(-1) for t in coord_entries], -1)

        extra = self.extra_transform \
                if self.extra_transform is not None else lambda i, x: x

        ret = {
            "ind":       ind,
            "img":       extra(ind, pack[0]),
            "label":     extra(ind, pack[1]),
            "img_pos":   extra(ind, pos_pack[0]),
            "label_pos": extra(ind, pos_pack[1]),
        }

        # Apply geometric augmentation to get img_aug + coord_aug
        if self.aug_geometric_transform is not None:
            aug_seed = np.random.randint(2147483647)
            # Apply same random transform to both image and coordinate grid
            self._set_seed(aug_seed)
            ret["img_aug"] = self.aug_geometric_transform(extra(ind, pack[0]))
            self._set_seed(aug_seed)
            coord_aug = self.aug_geometric_transform(
                coord.permute(2, 0, 1)  # [2, H, W] for torchvision transforms
            ).permute(1, 2, 0)          # back to [H, W, 2]
            ret["coord_aug"] = coord_aug
        else:
            ret["img_aug"] = extra(ind, pack[0])
            ret["coord_aug"] = coord

        ret["coord"] = coord

        if self.mask:
            ret["mask"] = pack[2]
            ret["mask_pos"] = pos_pack[2]

        if self.aug_photometric_transform is not None:
            ret["img_pos_aug"] = self.aug_photometric_transform(ret["img_pos"])

        return ret
class NegativeImageDataset(Dataset):
    """
    A simple dataset to load images from a directory recursively.
    Used to supply completely different "negative" images to the contrastive loss.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.dcm')):
                    self.image_paths.append(os.path.join(root, file))
        # Sort or shuffle if necessary, but dataloader shuffle will handle randomizing batches
        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Load using CHAOS dicom loader to preserve domain statistics (Hounsfield windowing)
        img = CHAOS._load_dicom_as_pil(img_path)
        if self.transform:
            img = self.transform(img)
        # We don't have labels or masks, just return the image
        return {"img": img}
