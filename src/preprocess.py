"""
preprocess.py — CHAOS Dataset Preprocessing Pipeline
=====================================================
Processes both CT and MR (T1DUAL InPhase, T2SPIR) modalities from the
CHAOS challenge dataset. Applies modality-specific preprocessing and
saves preprocessed images alongside their aligned ground truth masks
into a structured train/val split folder.

Output structure:
    preprocessed/
    ├── train/
    │   ├── images/   {MOD}_p{PID}_s{IDX:04d}.png
    │   └── labels/   {MOD}_p{PID}_s{IDX:04d}.png
    └── val/
        ├── images/
        └── labels/

CT pipeline:   HU conversion → Window [-150,250] → Rescale [0,255]
MR pipeline:   Percentile norm (1–99%) → Rescale [0,255]
Labels:        No transforms — saved as-is.
"""

import os
import numpy as np
import pydicom
import cv2
import SimpleITK as sitk
from PIL import Image
from tqdm.auto import tqdm


# ===========================================================================
# CONFIGURATION
# ===========================================================================

DATA_ROOT = os.path.join(os.path.dirname(__file__),
                         "pytorch_data_dir", "archive",
                         "CHAOS_Train_Sets", "Train_Sets")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__),
                          "preprocessed")

VAL_FRACTION = 0.2   # 80/20 patient-level split


# ===========================================================================
# PREPROCESSING HELPERS
# ===========================================================================

def preprocess_ct_slice(dcm_path):
    """CT preprocessing: HU → Window [-150,250] → Rescale [0,255] → CLAHE.

    Returns:
        final_img: preprocessed uint8 image (native resolution, no padding)
    """
    ds = pydicom.dcmread(dcm_path)
    hu = ds.pixel_array.astype(np.float32) * ds.RescaleSlope + ds.RescaleIntercept

    # A. Windowing [-150, 250]
    windowed = np.clip(hu, -150, 250)

    # B. Rescale to [0, 255] uint8
    rescaled = ((windowed + 150) / 400 * 255).astype(np.uint8)

    # C. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(rescaled)

    return enhanced


def preprocess_mri_slice(dcm_path):
    """MRI preprocessing: Percentile norm (1-99%) → N4 Bias Correction → Rescale [0,255] → CLAHE.

    Returns:
        final_img: preprocessed uint8 image (native resolution, no padding)
    """
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # A. Percentile normalization (1st–99th)
    lo, hi = np.percentile(arr, (1, 99))
    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)

    # B. N4 Bias Field Correction
    #    N4 requires positive values, so we work on the [0,1] normalized data
    try:
        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(sitk_img)
        arr = sitk.GetArrayFromImage(corrected)
        # Re-normalize to [0,1] after correction
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
    except Exception:
        pass  # If N4 fails on a slice, keep the uncorrected version

    # C. Rescale to uint8
    rescaled = (arr * 255).astype(np.uint8)

    # D. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(rescaled)

    return enhanced


# ===========================================================================
# SAMPLE COLLECTION
# ===========================================================================

def collect_samples(data_root):
    """Collect all (dcm_path, mask_path, modality, patient_id) tuples.

    CT:       sorted DICOM ↔ sorted liver_GT_NNN.png (index alignment)
    MR T1:    InPhase DICOM filename.dcm ↔ Ground/filename.png (name match)
    MR T2:    DICOM filename.dcm ↔ Ground/filename.png (name match)
    """
    samples = []

    # --- CT ---
    ct_root = os.path.join(data_root, "CT")
    if os.path.isdir(ct_root):
        for pid in sorted(os.listdir(ct_root)):
            pid_path = os.path.join(ct_root, pid)
            dicom_dir = os.path.join(pid_path, "DICOM_anon")
            ground_dir = os.path.join(pid_path, "Ground")

            # Skip non-patient folders (e.g. CHAOS, .DS_Store)
            if not os.path.isdir(dicom_dir) or not os.path.isdir(ground_dir):
                continue

            dcm_files = sorted([f for f in os.listdir(dicom_dir)
                                if f.lower().endswith(".dcm")])
            mask_files = sorted([f for f in os.listdir(ground_dir)
                                 if f.lower().endswith(".png")])

            # Index-based alignment (both sorted)
            for i, dcm in enumerate(dcm_files):
                mask = os.path.join(ground_dir, mask_files[i]) \
                       if i < len(mask_files) else None
                samples.append((
                    os.path.join(dicom_dir, dcm),
                    mask,
                    "CT",
                    pid
                ))

    # --- MR (T1DUAL InPhase + T2SPIR) ---
    mr_root = os.path.join(data_root, "MR")
    if os.path.isdir(mr_root):
        for pid in sorted(os.listdir(mr_root)):
            pid_path = os.path.join(mr_root, pid)
            if not os.path.isdir(pid_path):
                continue

            for seq, dcm_subdir in [("T1DUAL", os.path.join("DICOM_anon", "InPhase")),
                                     ("T2SPIR", "DICOM_anon")]:
                seq_path = os.path.join(pid_path, seq)
                dicom_dir = os.path.join(seq_path, dcm_subdir)
                ground_dir = os.path.join(seq_path, "Ground")

                if not os.path.isdir(dicom_dir) or not os.path.isdir(ground_dir):
                    continue

                dcm_files = sorted([f for f in os.listdir(dicom_dir)
                                    if f.lower().endswith(".dcm")])
                mask_files = {os.path.splitext(f)[0]: f
                              for f in os.listdir(ground_dir)
                              if f.lower().endswith(".png")}

                # Name-based alignment (DICOM stem == mask stem)
                for dcm in dcm_files:
                    stem = os.path.splitext(dcm)[0]
                    mask_path = os.path.join(ground_dir, mask_files[stem]) \
                                if stem in mask_files else None
                    samples.append((
                        os.path.join(dicom_dir, dcm),
                        mask_path,
                        seq,
                        pid
                    ))

    return samples


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def process_full_dataset():
    """Process the entire CHAOS dataset and save into preprocessed/ with
    an 80/20 patient-level train/val split.
    """
    print(f"📂 Scanning {DATA_ROOT} ...")
    all_samples = collect_samples(DATA_ROOT)
    print(f"   Found {len(all_samples)} total slices")

    if not all_samples:
        print("❌ ERROR: No samples found. Check DATA_ROOT path.")
        return

    # --- Patient-level split (per modality group) ---
    # Group patients by modality category (CT vs MR)
    ct_patients = sorted(set(s[3] for s in all_samples if s[2] == "CT"))
    mr_patients = sorted(set(s[3] for s in all_samples if s[2] in ("T1DUAL", "T2SPIR")))

    def pick_val(patients):
        n_val = max(1, int(VAL_FRACTION * len(patients)))
        return set(patients[-n_val:])

    val_ct = pick_val(ct_patients)
    val_mr = pick_val(mr_patients)

    print(f"   CT patients:  {len(ct_patients)} total → "
          f"{len(ct_patients) - len(val_ct)} train / {len(val_ct)} val  "
          f"(val: {sorted(val_ct)})")
    print(f"   MR patients:  {len(mr_patients)} total → "
          f"{len(mr_patients) - len(val_mr)} train / {len(val_mr)} val  "
          f"(val: {sorted(val_mr)})")

    # --- Create output directories ---
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    stats = {
        "train": {"images": 0, "labels": 0, "skipped": 0},
        "val":   {"images": 0, "labels": 0, "skipped": 0},
    }

    # --- Process all samples ---
    for idx, (dcm_path, mask_path, modality, pid) in enumerate(
            tqdm(all_samples, desc="Processing")):

        # Determine split
        if modality == "CT":
            split = "val" if pid in val_ct else "train"
        else:
            split = "val" if pid in val_mr else "train"

        fname = f"{modality}_p{pid}_s{idx:04d}.png"

        # Preprocess image
        try:
            if modality == "CT":
                img = preprocess_ct_slice(dcm_path)
            else:
                img = preprocess_mri_slice(dcm_path)
        except Exception as e:
            tqdm.write(f"  ⚠️ SKIP {dcm_path}: {e}")
            stats[split]["skipped"] += 1
            continue

        # Save preprocessed image
        Image.fromarray(img).save(
            os.path.join(OUTPUT_DIR, split, "images", fname))
        stats[split]["images"] += 1

        # Save label as-is (no spatial transforms needed without padding)
        if mask_path and os.path.exists(mask_path):
            label = np.array(Image.open(mask_path).convert("L"))
            # Convert bool masks to uint8
            if label.dtype == bool:
                label = label.astype(np.uint8) * 255
            Image.fromarray(label).save(
                os.path.join(OUTPUT_DIR, split, "labels", fname))
            stats[split]["labels"] += 1

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  ✅ PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Pipeline:")
    print(f"    CT:  HU → Window [-150,250] → Rescale [0,255] → CLAHE")
    print(f"    MR:  Percentile Norm (1-99%) → N4 Bias Correction → Rescale [0,255] → CLAHE")
    print(f"    Labels: Saved as-is (no spatial transforms)")
    for split in ("train", "val"):
        s = stats[split]
        print(f"  [{split.upper()}]  Images: {s['images']}  |  "
              f"Labels: {s['labels']}  |  Skipped: {s['skipped']}")
    print("=" * 60)


if __name__ == "__main__":
    process_full_dataset()