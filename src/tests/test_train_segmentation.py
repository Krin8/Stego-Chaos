from types import SimpleNamespace

import pytest
import torch

import train_segmentation


def test_get_class_labels_for_chaos_and_rejects_unknown_dataset():
    assert train_segmentation.get_class_labels("chaos") == [
        "background", "liver", "right kidney", "left kidney", "spleen"
    ]
    with pytest.raises(ValueError):
        train_segmentation.get_class_labels("unknown")


def test_train_import_is_hydra_guarded_and_add_spatial_coords_preserves_code():
    code = torch.zeros(2, 3, 3, 5, dtype=torch.float64)
    code[:, 0] = 1
    stub = SimpleNamespace(spatial_weight=2.0)
    actual = train_segmentation.LitUnsupervisedSegmenter._add_spatial_coords(stub, code)
    assert actual.shape == (2, 5, 3, 5)
    assert actual.dtype == code.dtype
    assert actual.device == code.device
    assert torch.equal(actual[:, :3], code)
    assert actual[0, 3, 0, 0].item() == -2
    assert actual[0, 3, -1, -1].item() == 2
    assert actual[0, 4, 0, 0].item() == -2
    assert actual[0, 4, -1, -1].item() == 2
    zero = train_segmentation.LitUnsupervisedSegmenter._add_spatial_coords(
        SimpleNamespace(spatial_weight=0.0), code
    )
    assert torch.equal(zero[:, 3:], torch.zeros_like(zero[:, 3:]))
