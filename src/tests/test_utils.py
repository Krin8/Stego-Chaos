import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

import utils


def test_one_hot_feats_has_expected_shape_dtype_and_values():
    labels = torch.tensor([[[0, 1, 2], [2, 1, 0]]])
    actual = utils.one_hot_feats(labels, 3)
    expected = torch.tensor(
        [[[[1, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]]],
        dtype=torch.float32,
    )
    assert actual.shape == (1, 3, 2, 3)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_resize_preserves_batch_channels_and_resizes_spatial_dimensions():
    x = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    actual = utils.resize(x, 4)
    assert actual.shape == (2, 3, 4, 4)


def test_shuffle_returns_a_permutation_of_rows():
    torch.manual_seed(0)
    x = torch.arange(20).reshape(5, 4)
    actual = utils.shuffle(x)
    assert actual.shape == x.shape
    assert sorted(actual.tolist()) == sorted(x.tolist())


def test_normalize_unnorm_round_trip_and_unnormalize_clones_input():
    x = torch.rand(3, 4, 5)
    normalized = utils.normalize(x)
    assert torch.allclose(utils.unnorm(normalized), x, atol=1e-6)
    source = normalized.clone()
    result = utils.UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(source)
    assert torch.equal(source, normalized)
    assert result.data_ptr() != source.data_ptr()


def test_prep_for_plot_resize_and_rescale_modes():
    x = torch.tensor(
        [
            [[-1.0, 0.0], [1.0, 2.0]],
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ]
    )
    no_resize = utils.prep_for_plot(x, resize=None)
    resized = utils.prep_for_plot(x, resize=(4, 4))
    assert no_resize.shape == (2, 2, 3)
    assert resized.shape == (4, 4, 3)
    assert no_resize.min().item() == 0.0
    assert no_resize.max().item() == 1.0

    expected = utils.unnorm(x.unsqueeze(0)).squeeze(0).permute(1, 2, 0)
    actual = utils.prep_for_plot(x, rescale=False)
    assert torch.equal(actual, expected)
    assert actual.min() < 0 or actual.max() > 1


def test_to_target_tensor_converts_l_image_to_int64_chw():
    image = Image.fromarray(np.array([[0, 3], [255, 7]], dtype=np.uint8), mode="L")
    actual = utils.ToTargetTensor()(image)
    assert actual.dtype == torch.int64
    assert actual.shape == (1, 2, 2)
    assert torch.equal(actual, torch.tensor([[[0, 3], [255, 7]]], dtype=torch.int64))


def test_prep_args_normalizes_supported_cli_forms_and_rejects_bare_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["p", "a=1"])
    utils.prep_args()
    assert sys.argv == ["p", "a=1"]

    monkeypatch.setattr(sys, "argv", ["p", "--foo", "bar"])
    utils.prep_args()
    assert sys.argv == ["p", "foo=bar"]

    monkeypatch.setattr(sys, "argv", ["p", "oops"])
    with pytest.raises(ValueError):
        utils.prep_args()


@pytest.mark.parametrize("crop_type", ["center", "random"])
def test_get_transform_crops_image_and_label_to_requested_resolution(crop_type):
    image = Image.fromarray(np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3))
    label = Image.fromarray(np.array([[0, 63, 126], [189, 252, 255]], dtype=np.uint8), mode="L")
    image_out = utils.get_transform(6, False, crop_type)(image)
    label_out = utils.get_transform(6, True, crop_type)(label)
    assert image_out.shape == (3, 6, 6)
    assert label_out.shape == (1, 6, 6)
    assert label_out.dtype == torch.int64


def test_get_transform_none_resizes_and_unknown_crop_type_raises():
    image = Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8))
    label = Image.fromarray(np.zeros((10, 12), dtype=np.uint8), mode="L")
    assert utils.get_transform(6, False, None)(image).shape == (3, 6, 6)
    assert utils.get_transform(6, True, None)(label).shape == (1, 6, 6)
    with pytest.raises(ValueError):
        utils.get_transform(6, False, "diagonal")


def test_identity_transform_returns_same_object():
    value = object()
    assert utils.identity_transform(value) is value


def test_load_model_unknown_type_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        utils.load_model("nonexistent", str(tmp_path))


@pytest.mark.parametrize("shape", [(3,), (2, 2)])
def test_remove_axes_clears_ticks_for_one_and_two_dimensional_axes(shape):
    if len(shape) == 1:
        _, axes = plt.subplots(1, 3)
    else:
        _, axes = plt.subplots(2, 2)
    utils.remove_axes(axes)
    axes_iter = axes.flat if hasattr(axes, "flat") else axes
    for ax in axes_iter:
        assert ax.get_xticks().size == 0
        assert ax.get_yticks().size == 0
    plt.close("all")


class _ImageWriter:
    def __init__(self):
        self.calls = []

    def add_image(self, name, tensor, step):
        self.calls.append((name, tensor, step))


def test_add_plot_supports_single_writer_and_list_of_writers():
    plt.figure()
    plt.plot([0, 1], [1, 0])
    writer = _ImageWriter()
    utils.add_plot(writer, "plot/name", 7)
    assert len(writer.calls) == 1
    name, tensor, step = writer.calls[0]
    assert name == "plot/name"
    assert step == 7
    assert tensor.ndim == 3
    assert tensor.shape[0] == 3
    assert tensor.dtype == torch.float32

    plt.figure()
    plt.plot([0, 1], [0, 1])
    writer_a, writer_b = _ImageWriter(), _ImageWriter()
    utils.add_plot([writer_a, writer_b], "other", 8)
    assert writer_a.calls[0][0] == writer_b.calls[0][0] == "other"
    assert writer_a.calls[0][2] == writer_b.calls[0][2] == 8


def test_unsupervised_metrics_perfect_predictions_return_prefixed_100_scores():
    metrics = utils.UnsupervisedMetrics("x/", 3, 0, False)
    target = torch.tensor([[0, 1, 2], [2, 1, 0]])
    metrics.update(target, target)
    result = metrics.compute()
    assert result == {"x/mIoU": 100.0, "x/Accuracy": 100.0}


def test_unsupervised_metrics_hungarian_recovers_fixed_permutation():
    metrics = utils.UnsupervisedMetrics("p/", 3, 0, True)
    target = torch.tensor([[0, 1, 2], [2, 1, 0]])
    permutation = torch.tensor([[1, 2, 0], [0, 2, 1]])
    metrics.update(permutation, target)
    assert metrics.compute()["p/mIoU"] == 100.0


def test_unsupervised_metrics_excludes_invalid_targets_and_predictions():
    metrics = utils.UnsupervisedMetrics("p/", 2, 0, False)
    preds = torch.tensor([[0, 1, 1, 3]])
    target = torch.tensor([[0, 1, -1, 0]])
    metrics.update(preds, target)
    assert metrics.stats.sum().item() == 2


def test_unsupervised_metrics_extra_clusters_has_finite_scores_and_expected_histogram():
    metrics = utils.UnsupervisedMetrics("p/", 2, 1, True)
    metrics.update(torch.tensor([[0, 1, 2, 0]]), torch.tensor([[0, 1, 1, 0]]))
    result = metrics.compute()
    assert all(np.isfinite(value) for value in result.values())
    assert metrics.histogram.shape == (3, 3)


def test_unsupervised_metrics_map_clusters_matches_assignments_and_extra_branch():
    metrics = utils.UnsupervisedMetrics("p/", 2, 0, True)
    target = torch.tensor([[0, 1, 0, 1]])
    preds = torch.tensor([[1, 0, 1, 0]])
    metrics.update(preds, target)
    metrics.compute()
    mapped = metrics.map_clusters(preds)
    assert torch.equal(mapped, target)
    extra = utils.UnsupervisedMetrics("p/", 2, 1, True)
    extra.update(torch.tensor([[0, 1, 2, 0]]), torch.tensor([[0, 1, 1, 0]]))
    extra.compute()
    mapped_extra = extra.map_clusters(torch.tensor([[0, 1, 2]]))
    assert mapped_extra.shape == (1, 3)


def test_unsupervised_metrics_empty_update_returns_zero_scores():
    metrics = utils.UnsupervisedMetrics("p/", 2, 0, False)
    metrics.update(torch.tensor([[-1, -1]]), torch.tensor([[-1, -1]]))
    result = metrics.compute()
    assert result["p/mIoU"] == 0.0
    assert result["p/Accuracy"] == 0.0
    assert all(np.isfinite(value) for value in result.values())


def test_flexible_collate_tensor_and_unequal_shape_fallback():
    actual = utils.flexible_collate([torch.ones(2), torch.zeros(2)])
    assert torch.equal(actual, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    unequal = [torch.ones(2), torch.zeros(3)]
    assert utils.flexible_collate(unequal) == unequal


def test_flexible_collate_numpy_arrays_scalars_and_numeric_types():
    arrays = utils.flexible_collate([np.ones((2,)), np.zeros((2,))])
    assert torch.equal(arrays, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    assert torch.equal(utils.flexible_collate([np.int32(1), np.int32(2)]), torch.tensor([1, 2]))
    floats = utils.flexible_collate([1.0, 2.0])
    ints = utils.flexible_collate([1, 2])
    assert floats.dtype == torch.float64
    assert ints.dtype == torch.int64


def test_flexible_collate_strings_dicts_and_sequences():
    assert utils.flexible_collate(["a", "b"]) == ["a", "b"]
    result = utils.flexible_collate([{"x": torch.tensor(1), "y": torch.tensor([2])},
                                     {"x": torch.tensor(3), "y": torch.tensor([4])}])
    assert torch.equal(result["x"], torch.tensor([1, 3]))
    assert torch.equal(result["y"], torch.tensor([[2], [4]]))
    transposed = utils.flexible_collate([[1, 2], [3, 4]])
    assert torch.equal(transposed[0], torch.tensor([1, 3]))
    assert torch.equal(transposed[1], torch.tensor([2, 4]))
    with pytest.raises(RuntimeError):
        utils.flexible_collate([[1], [2, 3]])
    with pytest.raises(TypeError):
        utils.flexible_collate([object()])
