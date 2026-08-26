from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from conftest import write_dicom
import data as data_module
from data import (
    CHAOS,
    ContrastiveSegDataset,
    MaterializedDataset,
    NegativeImageDataset,
    bit_get,
    create_chaos_colormap,
    create_pascal_label_colormap,
)
from utils import get_transform


def _train_root(root):
    path = root / "archive" / "CHAOS_Train_Sets" / "Train_Sets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _add_ct_patient(root, pid, count=1, masks=None):
    patient = _train_root(root) / "CT" / str(pid)
    dicom_dir, ground_dir = patient / "DICOM_anon", patient / "Ground"
    dicom_dir.mkdir(parents=True, exist_ok=True)
    ground_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        write_dicom(
            dicom_dir / f"slice_{index:03d}.dcm",
            np.full((16, 16), index + 1, dtype=np.int16),
            intercept=0,
        )
    for index, value in enumerate(masks if masks is not None else [0] * count):
        Image.fromarray(np.full((16, 16), value, dtype=np.uint8), mode="L").save(
            ground_dir / f"mask_{index:03d}.png"
        )


def _add_mr_patient(root, pid, seq="T1DUAL", count=1, names=None):
    patient = _train_root(root) / "MR" / str(pid) / seq
    dcm_dir = patient / "DICOM_anon" / "InPhase" if seq == "T1DUAL" else patient / "DICOM_anon"
    ground_dir = patient / "Ground"
    dcm_dir.mkdir(parents=True, exist_ok=True)
    ground_dir.mkdir(parents=True, exist_ok=True)
    names = names or [f"slice_{index:03d}" for index in range(count)]
    for index, name in enumerate(names):
        write_dicom(dcm_dir / f"{name}.dcm", np.full((16, 16), index + 1, dtype=np.int16), intercept=0)
        Image.fromarray(np.full((16, 16), 63, dtype=np.uint8), mode="L").save(
            ground_dir / f"{name}.png"
        )


def _simple_transforms(res=32):
    return get_transform(res, False, None), get_transform(res, True, None)


def test_bit_get_and_colormaps():
    assert bit_get(0b1010, 1) == 1
    assert bit_get(0b1010, 0) == 0
    assert bit_get(255, 7) == 1
    pascal = create_pascal_label_colormap()
    assert pascal.shape == (512, 3)
    assert np.array_equal(pascal[0], [0, 0, 0])
    assert np.array_equal(pascal[1], [128, 0, 0])
    assert pascal.dtype == int
    assert pascal.min() >= 0 and pascal.max() <= 255
    chaos = create_chaos_colormap()
    assert chaos.dtype == np.uint8
    assert np.array_equal(
        chaos,
        np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 128, 255], [255, 255, 0]], dtype=np.uint8),
    )


def test_load_mask_maps_ct_and_mri_pixels_and_preserves_l_mode(tmp_path):
    path = tmp_path / "mask.png"
    raw = np.array([[0, 255, 7], [63, 126, 189], [252, 1, 255]], dtype=np.uint8)
    Image.fromarray(raw, mode="L").save(path)
    ct = np.array(CHAOS._load_mask(str(path), "CT"))
    mri = np.array(CHAOS._load_mask(str(path), "T1DUAL"))
    assert CHAOS._load_mask(str(path), "CT").mode == "L"
    assert np.array_equal(ct, [[0, 1, 255], [255, 255, 255], [255, 255, 1]])
    assert np.array_equal(mri, [[0, 255, 255], [1, 2, 3], [4, 255, 255]])


def test_load_dicom_as_pil_applies_fixed_abdominal_window_and_metadata_defaults(dicom_factory):
    path = dicom_factory(
        "window.dcm",
        shape=(1, 5),
        fill=np.array([[-200, -160, 40, 240, 300]], dtype=np.int16),
        slope=1,
        intercept=0,
    )
    actual = np.array(CHAOS._load_dicom_as_pil(str(path)))
    assert actual.shape == (1, 5, 3)
    assert np.array_equal(actual[0, 0], [0, 0, 0])
    assert np.array_equal(actual[0, 1], [0, 0, 0])
    assert np.array_equal(actual[0, 3], [255, 255, 255])
    assert np.array_equal(actual[0, 4], [255, 255, 255])
    assert 127 <= actual[0, 2, 0] <= 128

    no_tags = dicom_factory("no_tags.dcm", shape=(1, 1), fill=np.array([[40]], dtype=np.int16))
    import pydicom

    ds = pydicom.dcmread(no_tags)
    del ds.RescaleSlope
    del ds.RescaleIntercept
    ds.save_as(no_tags)
    defaulted = np.array(CHAOS._load_dicom_as_pil(str(no_tags)))
    assert 127 <= defaulted[0, 0, 0] <= 128


def test_chaos_rejects_bad_modality_and_image_set(tmp_path):
    with pytest.raises(AssertionError):
        CHAOS(str(tmp_path), "bad", "all", None, None)
    with pytest.raises(AssertionError):
        CHAOS(str(tmp_path), "CT", "bad", None, None)


def test_chaos_collects_ct_mr_modalities_and_missing_masks(tmp_path):
    _add_ct_patient(tmp_path, 1, count=2, masks=[0])
    _add_mr_patient(tmp_path, 2, "T1DUAL", count=1)
    _add_mr_patient(tmp_path, 3, "T2SPIR", count=1)
    (_train_root(tmp_path) / "CT" / "4").mkdir(parents=True)
    (_train_root(tmp_path) / "MR" / "5" / "T1DUAL").mkdir(parents=True)

    ct = CHAOS(str(tmp_path), "CT", "all", None, None)
    assert len(ct.samples) == 2
    assert ct.samples[0][2] == "CT"
    assert ct.samples[0][1] is not None
    assert ct.samples[1][1] is None
    t1 = CHAOS(str(tmp_path), "T1DUAL", "all", None, None)
    t2 = CHAOS(str(tmp_path), "T2SPIR", "all", None, None)
    assert len(t1) == len(t2) == 1
    assert "InPhase" in t1.samples[0][0]
    assert t2.samples[0][2] == "T2SPIR"
    all_data = CHAOS(str(tmp_path), "all", "all", None, None)
    assert {sample[2] for sample in all_data.samples} == {"CT", "T1DUAL", "T2SPIR"}


def test_chaos_patient_level_train_val_split_is_disjoint_and_complete(tmp_path):
    for pid in range(1, 6):
        _add_ct_patient(tmp_path, pid)
    transforms = _simple_transforms(16)
    all_data = CHAOS(str(tmp_path), "CT", "all", *transforms)
    train = CHAOS(str(tmp_path), "CT", "train", *transforms)
    val = CHAOS(str(tmp_path), "CT", "val", *transforms)
    all_ids = {sample[3] for sample in all_data.samples}
    train_ids = {sample[3] for sample in train.samples}
    val_ids = {sample[3] for sample in val.samples}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == all_ids
    assert len(val_ids) == max(1, int(0.2 * len(all_ids)))


def test_chaos_mri_getitem_returns_contract_and_maps_ignore_to_minus_one(tmp_path):
    _add_mr_patient(tmp_path, 1, "T1DUAL", count=1)
    mask_path = _train_root(tmp_path) / "MR" / "1" / "T1DUAL" / "Ground" / "slice_000.png"
    raw = np.array([[63, 17] * 8] * 16, dtype=np.uint8)
    Image.fromarray(raw, mode="L").save(mask_path)
    transform, target_transform = _simple_transforms(32)
    ds = CHAOS(str(tmp_path), "T1DUAL", "all", transform, target_transform)
    image, label, valid_mask = ds[0]
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label.shape == (32, 32)
    assert label.dtype == torch.int64
    assert valid_mask.shape == (32, 32)
    assert -1 in label
    assert 255 not in label

    ds.samples[0] = (ds.samples[0][0], None, ds.samples[0][2], ds.samples[0][3])
    _, no_mask_label, _ = ds[0]
    assert torch.equal(no_mask_label, torch.full((32, 32), -1, dtype=torch.long))


def test_chaos_preprocessed_collection_filters_names_and_getitem_contract(tmp_path):
    preprocessed = tmp_path / "preprocessed"
    image_dir, label_dir = preprocessed / "train" / "images", preprocessed / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for name, value in (("CT_p1_s0000.png", 20), ("T1DUAL_p2_s0000.png", 30), ("MR_p3_s0000.png", 40)):
        Image.fromarray(np.full((16, 16), value, dtype=np.uint8), mode="L").save(image_dir / name)
    Image.fromarray(np.full((16, 16), 255, dtype=np.uint8), mode="L").save(label_dir / "CT_p1_s0000.png")
    Image.fromarray(np.full((16, 16), 63, dtype=np.uint8), mode="L").save(label_dir / "T1DUAL_p2_s0000.png")
    (image_dir / "not_png.txt").write_text("ignored")
    (image_dir / "malformed.png").write_bytes(b"ignored")
    transform, target_transform = _simple_transforms(16)
    ds = CHAOS(str(tmp_path / "data"), "CT", "train", transform, target_transform, use_preprocessed_data=True)
    assert len(ds) == 1
    assert ds.samples[0][2] == "CT"
    assert ds.samples[0][1] is not None
    image, label, valid_mask = ds[0]
    assert image.shape == (3, 16, 16)
    assert label.shape == (16, 16)
    assert valid_mask.shape == (16, 16)

    missing = image_dir / "CT_p9_s0001.png"
    Image.fromarray(np.full((16, 16), 1, dtype=np.uint8), mode="L").save(missing)
    ds_missing = CHAOS(str(tmp_path / "data"), "CT", "train", transform, target_transform, use_preprocessed_data=True)
    assert any(sample[3] == "p9" and sample[1] is None for sample in ds_missing.samples)


class _TinyDataset(Dataset):
    def __init__(self):
        self.items = [("a", 1), ("b", 2), ("c", 3)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_materialized_dataset_matches_underlying_items():
    source = _TinyDataset()
    materialized = MaterializedDataset(source)
    assert len(materialized) == len(source)
    assert [materialized[i] for i in range(len(source))] == source.items


def test_negative_image_dataset_walks_sorted_dicom_paths_and_transforms(tmp_path):
    (tmp_path / "z").mkdir()
    write_dicom(tmp_path / "z" / "b.dcm", np.ones((8, 8), dtype=np.int16), intercept=0)
    write_dicom(tmp_path / "a.dcm", np.ones((8, 8), dtype=np.int16), intercept=0)
    (tmp_path / "ignore.png").write_bytes(b"not dicom")
    dataset = NegativeImageDataset(str(tmp_path))
    assert dataset.image_paths == sorted(dataset.image_paths)
    assert len(dataset) == 2
    assert dataset[0]["img"].mode == "RGB"
    transformed = NegativeImageDataset(str(tmp_path), transform=lambda image: torch.from_numpy(np.array(image)))
    assert transformed[0]["img"].shape == (8, 8, 3)


def _preprocessed_contrastive_tree(root):
    image_dir = root / "preprocessed" / "train" / "images"
    label_dir = root / "preprocessed" / "train" / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("CT_p1_s0000.png", 20),
        ("CT_p1_s0001.png", 40),
        ("CT_p2_s0000.png", 60),
    ):
        Image.fromarray(np.full((16, 16), value, dtype=np.uint8), mode="L").save(image_dir / name)
        Image.fromarray(np.full((16, 16), 255, dtype=np.uint8), mode="L").save(label_dir / name)


def test_contrastive_seg_dataset_keys_coords_positive_candidates_and_transforms(tmp_path, monkeypatch):
    _preprocessed_contrastive_tree(tmp_path)
    transform, target_transform = _simple_transforms(16)
    cfg = SimpleNamespace(chaos_modality="CT", chaos_n_classes=2, use_preprocessed_data=True)
    ds = ContrastiveSegDataset(
        str(tmp_path / "data"), "chaos", None, "train", transform, target_transform, cfg,
        mask=True, pos_images=False,
    )
    item = ds[0]
    assert set(item) == {"ind", "img", "label", "img_pos", "label_pos", "img_aug", "coord_aug", "coord", "mask", "mask_pos"}
    assert item["coord"].shape == (16, 16, 2)
    assert item["coord"][..., 0].min() == -1 and item["coord"][..., 0].max() == 1
    assert item["coord"][..., 1].min() == -1 and item["coord"][..., 1].max() == 1
    assert torch.equal(item["img"], item["img_pos"])
    assert ds._pos_candidates[0] == [1]
    assert 0 not in ds._pos_candidates[0]
    assert ds._pos_candidates[2] == [2]

    positive = ContrastiveSegDataset(
        str(tmp_path / "data"), "chaos", None, "train", transform, target_transform, cfg,
        pos_images=True,
    )
    chosen = {}
    def choose(candidates):
        chosen["value"] = candidates[0]
        return candidates[0]
    monkeypatch.setattr(data_module.random, "choice", choose)
    positive_item = positive[0]
    assert not torch.equal(positive_item["img"], positive_item["img_pos"])
    assert chosen["value"] in positive._pos_candidates[0]
    marker = lambda index, value: value * 0
    photometric = ContrastiveSegDataset(
        str(tmp_path / "data"), "chaos", None, "train", transform, target_transform, cfg,
        extra_transform=marker, aug_photometric_transform=lambda image: image + 1,
    )
    photometric_item = photometric[0]
    assert set(photometric_item) == {
        "ind", "img", "label", "img_pos", "label_pos", "img_aug", "coord_aug", "coord", "img_pos_aug"
    }
    assert torch.equal(photometric_item["img"], torch.zeros_like(photometric_item["img"]))
    assert torch.equal(photometric_item["label"], torch.zeros_like(photometric_item["label"]))
    assert torch.equal(photometric_item["img_pos_aug"], torch.ones_like(photometric_item["img_pos_aug"]))
    assert len(ds) == len(ds.dataset)
