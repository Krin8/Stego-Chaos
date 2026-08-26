import numpy as np
import pytest
from PIL import Image

import preprocess
from conftest import write_dicom


def _root(root):
    path = root / "CT"
    path.mkdir(parents=True, exist_ok=True)
    (root / "MR").mkdir(parents=True, exist_ok=True)
    return root


def _ct_patient(root, pid, names=("a.dcm",), masks=("m.png",), corrupt=None):
    patient = root / "CT" / str(pid)
    dcm_dir, ground_dir = patient / "DICOM_anon", patient / "Ground"
    dcm_dir.mkdir(parents=True, exist_ok=True)
    ground_dir.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        if corrupt is not None and name == corrupt:
            (dcm_dir / name).write_bytes(b"corrupt")
        else:
            write_dicom(dcm_dir / name, np.full((8, 8), index + 1, dtype=np.int16), intercept=0)
    for name in masks:
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(ground_dir / name)


def _mr_patient(root, pid, seq, names=("a.dcm",), masks=("a.png",)):
    patient = root / "MR" / str(pid) / seq
    dcm_dir = patient / "DICOM_anon" / "InPhase" if seq == "T1DUAL" else patient / "DICOM_anon"
    ground_dir = patient / "Ground"
    dcm_dir.mkdir(parents=True, exist_ok=True)
    ground_dir.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        write_dicom(dcm_dir / name, np.full((8, 8), index + 2, dtype=np.int16), intercept=0)
    for name in masks:
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(ground_dir / name)


def test_preprocess_ct_slice_windowed_uint8_and_constant_below_window(dicom_factory):
    path = dicom_factory(
        "ct.dcm",
        shape=(4, 4),
        fill=np.array([[-300, -150, 250, 400]] * 4, dtype=np.int16),
        slope=2,
        intercept=-100,
    )
    actual = preprocess.preprocess_ct_slice(str(path))
    assert actual.dtype == np.uint8
    assert actual.shape == (4, 4)
    assert actual.min() >= 0 and actual.max() <= 255
    constant = dicom_factory("constant.dcm", shape=(4, 4), fill=np.full((4, 4), -500), intercept=0)
    constant_result = preprocess.preprocess_ct_slice(str(constant))
    assert np.all(constant_result == constant_result[0, 0])


def test_preprocess_mri_slice_constant_branch_and_n4_fallback(dicom_factory, monkeypatch):
    path = dicom_factory("mri.dcm", shape=(4, 4), fill=np.arange(16).reshape(4, 4), intercept=0)
    actual = preprocess.preprocess_mri_slice(str(path))
    assert actual.dtype == np.uint8
    assert actual.shape == (4, 4)
    assert actual.min() >= 0 and actual.max() <= 255

    constant = dicom_factory("constant_mri.dcm", shape=(4, 4), fill=np.full((4, 4), 7), intercept=0)
    constant_result = preprocess.preprocess_mri_slice(str(constant))
    # CLAHE maps a flat image to an arbitrary constant, so pin uniformity instead.
    assert constant_result.dtype == np.uint8
    assert constant_result.shape == (4, 4)
    assert not np.isnan(constant_result).any()
    assert np.unique(constant_result).size == 1

    class RaisingN4:
        def __init__(self):
            raise RuntimeError("N4 unavailable")

    monkeypatch.setattr(preprocess.sitk, "N4BiasFieldCorrectionImageFilter", RaisingN4)
    fallback = preprocess.preprocess_mri_slice(str(path))
    assert fallback.dtype == np.uint8
    assert fallback.shape == (4, 4)


def test_collect_samples_aligns_ct_by_index_and_mr_by_stem(tmp_path):
    root = _root(tmp_path)
    _ct_patient(root, 1, names=("b.dcm", "a.dcm", "extra.dcm"), masks=("m0.png", "m1.png"))
    _mr_patient(root, 2, "T1DUAL", names=("a.dcm", "b.dcm"), masks=("b.png",))
    _mr_patient(root, 3, "T2SPIR", names=("same.dcm", "missing.dcm"), masks=("same.png",))
    samples = preprocess.collect_samples(str(root))
    ct = [sample for sample in samples if sample[2] == "CT"]
    t1 = [sample for sample in samples if sample[2] == "T1DUAL"]
    t2 = [sample for sample in samples if sample[2] == "T2SPIR"]
    assert [sample[0].split("/")[-1] for sample in ct] == ["a.dcm", "b.dcm", "extra.dcm"]
    assert ct[-1][1] is None
    assert t1[0][1] is None and t1[1][1].endswith("b.png")
    t2_by_name = {sample[0].split("/")[-1]: sample[1] for sample in t2}
    assert t2_by_name["same.dcm"].endswith("same.png")
    assert t2_by_name["missing.dcm"] is None

    (root / "CT" / ".DS_Store").write_text("ignored")
    (root / "CT" / "4").mkdir()
    (root / "MR" / "5" / "T1DUAL").mkdir(parents=True)
    assert len(preprocess.collect_samples(str(root))) == len(samples)
    assert preprocess.collect_samples(str(tmp_path / "missing")) == []
    assert preprocess.collect_samples(str(tmp_path / "empty")) == []


def test_process_full_dataset_writes_split_outputs_and_skips_corrupt_slice(tmp_path, monkeypatch):
    data_root = _root(tmp_path / "data")
    _ct_patient(data_root, 1, names=("a.dcm",))
    _ct_patient(data_root, 2, names=("a.dcm", "bad.dcm"), corrupt="bad.dcm")
    _mr_patient(data_root, 1, "T1DUAL", names=("a.dcm",))
    _mr_patient(data_root, 2, "T2SPIR", names=("a.dcm",))
    output = tmp_path / "output"
    monkeypatch.setattr(preprocess, "DATA_ROOT", str(data_root))
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", str(output))
    preprocess.process_full_dataset()
    image_paths = sorted(output.glob("*/images/*.png"))
    label_paths = sorted(output.glob("*/labels/*.png"))
    assert len(image_paths) == 4
    assert len(label_paths) == 4
    assert {path.parent.parent.name for path in image_paths} == {"train", "val"}
    assert all(path.name.startswith(("CT_p", "T1DUAL_p", "T2SPIR_p")) for path in image_paths)
    assert all("_s" in path.name and path.stem.split("_s")[1].isdigit() for path in image_paths)

    empty_output = tmp_path / "empty_output"
    monkeypatch.setattr(preprocess, "DATA_ROOT", str(tmp_path / "not_there"))
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", str(empty_output))
    assert preprocess.process_full_dataset() is None
    assert not empty_output.exists()
