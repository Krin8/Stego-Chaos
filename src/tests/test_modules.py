from types import SimpleNamespace

import pytest
import torch
from torch import nn

import modules


def test_norm_normalizes_rows_and_zero_rows_are_finite():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    actual = modules.norm(x)
    assert torch.allclose(actual[0].norm(), torch.tensor(1.0))
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[1], torch.zeros(2))


def test_average_norm_matches_hand_computed_value():
    x = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    expected = x / ((5.0 + 2.0) / 2)
    assert torch.allclose(modules.average_norm(x), expected)


def test_tensor_correlation_matches_explicit_reference():
    a = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
    b = torch.tensor([[[[2.0, 1.0], [0.0, 3.0]], [[4.0, 3.0], [2.0, 1.0]]]])
    actual = modules.tensor_correlation(a, b)
    expected = torch.empty(1, 2, 2, 2, 2)
    for n in range(1):
        for h in range(2):
            for w in range(2):
                for i in range(2):
                    for j in range(2):
                        expected[n, h, w, i, j] = (a[n, :, h, w] * b[n, :, i, j]).sum()
    assert actual.shape == (1, 2, 2, 2, 2)
    assert torch.equal(actual, expected)


def test_sample_corner_coordinate_returns_corner_pixel_with_align_corners():
    x = torch.arange(1, 10, dtype=torch.float32).reshape(1, 1, 3, 3)
    coords = torch.tensor([[[[-1.0, -1.0]]]])
    actual = modules.sample(x, coords)
    assert actual.shape == (1, 1, 1, 1)
    assert actual.item() == x[0, 0, 0, 0].item()


def test_super_perm_is_in_range_and_has_no_fixed_points():
    for size in (2, 5, 10):
        torch.manual_seed(size)
        perm = modules.super_perm(size, torch.device("cpu"))
        assert perm.shape == (size,)
        assert ((perm >= 0) & (perm < size)).all()
        assert not (perm == torch.arange(size)).any()


def test_sample_nonzero_locations_stays_in_nonzero_quadrant_and_handles_empty():
    salience = torch.zeros(1, 8, 8)
    salience[:, :4, :4] = 1
    torch.manual_seed(0)
    coords = modules.sample_nonzero_locations(salience, [1, 20, 20, 2])
    assert coords.shape == (1, 20, 20, 2)
    assert (coords[..., 0] >= -1).all() and (coords[..., 0] < 0).all()
    assert (coords[..., 1] >= -1).all() and (coords[..., 1] < 0).all()

    empty = modules.sample_nonzero_locations(torch.zeros(1, 8, 8), [1, 3, 4, 2])
    assert empty.shape == (1, 3, 4, 2)
    assert (empty >= -1).all() and (empty <= 1).all()


def test_lambda_layer_applies_lambda():
    layer = modules.LambdaLayer(lambda x: x + 3)
    assert torch.equal(layer(torch.tensor([2])), torch.tensor([5]))


def test_cluster_lookup_init_from_copies_truncates_and_zeroes_tail():
    lookup = modules.ClusterLookup(4, 2)
    lookup.clusters.data.fill_(9)
    narrow = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    lookup.init_from(narrow)
    assert torch.equal(lookup.clusters[0, :2], narrow[0])
    assert torch.equal(lookup.clusters[1, :2], narrow[1])
    assert torch.equal(lookup.clusters[:, 2:], torch.zeros(2, 2))

    wide_lookup = modules.ClusterLookup(2, 2)
    wide_lookup.clusters.data.fill_(9)
    wide_lookup.init_from(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert torch.equal(wide_lookup.clusters, torch.tensor([[1.0, 2.0], [4.0, 5.0]]))


def test_cluster_lookup_reset_parameters_changes_weights():
    lookup = modules.ClusterLookup(3, 2)
    before = lookup.clusters.detach().clone()
    torch.manual_seed(1)
    lookup.reset_parameters()
    assert not torch.equal(before, lookup.clusters)


def test_cluster_lookup_forward_one_hot_and_exact_loss():
    lookup = modules.ClusterLookup(2, 2)
    lookup.clusters.data.copy_(torch.eye(2))
    x = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    loss, probs = lookup(x, alpha=None)
    assert probs.shape == (1, 2, 1, 2)
    assert torch.equal(probs.sum(1), torch.ones(1, 1, 2))
    assert torch.equal(probs.max(1).values, torch.ones(1, 1, 2))
    normed_clusters = torch.nn.functional.normalize(lookup.clusters, dim=1)
    normed_features = torch.nn.functional.normalize(x, dim=1)
    inner = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
    expected_probs = torch.nn.functional.one_hot(torch.argmax(inner, dim=1), 2).permute(0, 3, 1, 2).float()
    expected_loss = -(expected_probs * inner).sum(1).mean()
    avg_probs = expected_probs.mean(dim=(0, 2, 3))
    entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
    expected_loss -= 1.5 * entropy
    assert torch.allclose(loss, expected_loss)


def test_cluster_lookup_forward_softmax_and_log_probs():
    lookup = modules.ClusterLookup(2, 2)
    lookup.clusters.data.copy_(torch.eye(2))
    x = torch.tensor([[[[1.0]], [[0.5]]]])
    _, probs = lookup(x, alpha=10.0)
    assert torch.allclose(probs.sum(1), torch.ones(1, 1, 1))
    assert ((probs > 0) & (probs < 1)).all()
    log_probs = lookup(x, alpha=10.0, log_probs=True)
    assert torch.allclose(log_probs.exp().sum(1), torch.ones(1, 1, 1))


def test_resize_and_classify_returns_log_probabilities_at_input_size():
    model = modules.ResizeAndClassify(2, 2, 3)
    x = torch.randn(1, 2, 2, 2)
    actual = model(x)
    assert actual.shape == (1, 3, 2, 2)
    assert torch.allclose(actual.exp().sum(1), torch.ones(1, 2, 2))


def test_double_conv_preserves_spatial_shape():
    model = modules.DoubleConv(2, 3)
    assert model(torch.randn(2, 2, 5, 7)).shape == (2, 3, 5, 7)


def test_decoder_output_equals_linear_plus_nonlinear():
    model = modules.Decoder(2, 3)
    x = torch.randn(1, 2, 4, 4)
    assert torch.equal(model(x), model.linear(x) + model.nonlinear(x))
    assert model(x).shape == (1, 3, 4, 4)


def test_net_with_activations_resolves_negative_layer_numbers():
    model = nn.Sequential(nn.Identity(), nn.ReLU(), nn.Linear(3, 3), nn.Sigmoid())
    wrapped = modules.NetWithActivations(model, [1, -1])
    x = torch.tensor([[-1.0, 0.5, 2.0]])
    expected = {}
    for index, layer in enumerate(model):
        x = layer(x)
        if index in (1, 3):
            expected[index] = x
    actual = wrapped(torch.tensor([[-1.0, 0.5, 2.0]]))
    assert set(actual) == {1, 3}
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[3], expected[3])


def _corr_cfg(**overrides):
    values = dict(
        feature_samples=4,
        neg_samples=2,
        use_salience=False,
        pointwise=False,
        zero_clamp=False,
        stabalize=False,
        pos_intra_shift=0.0,
        pos_inter_shift=0.0,
        neg_inter_shift=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_contrastive_correlation_standard_scale_has_zero_mean_unit_std():
    loss = modules.ContrastiveCorrelationLoss(_corr_cfg())
    actual = loss.standard_scale(torch.arange(1.0, 10.0))
    assert torch.allclose(actual.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(actual.std(), torch.tensor(1.0), atol=1e-6)


def test_contrastive_correlation_helper_clamping_and_pointwise_modes():
    f = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    c = f.clone()
    opposite = -c
    unclamped = modules.ContrastiveCorrelationLoss(_corr_cfg())
    zero_clamped = modules.ContrastiveCorrelationLoss(_corr_cfg(zero_clamp=True))
    loss_unclamped, cd = unclamped.helper(f, f, c, opposite, 0.0)
    loss_zero, zero_cd = zero_clamped.helper(f, f, c, opposite, 0.0)
    assert torch.equal(cd, zero_cd)
    assert torch.all(loss_zero == 0)
    assert (loss_unclamped > 0).any()
    assert (loss_unclamped == 0).any()

    stable = modules.ContrastiveCorrelationLoss(_corr_cfg(stabalize=True))
    _, stable_cd = stable.helper(f, f, c, c, 0.0)
    _, regular_cd = unclamped.helper(f, f, c, c, 0.0)
    assert torch.equal(stable_cd, regular_cd)
    stable_loss, _ = stable.helper(f, f, c, c, 0.0)
    regular_loss, _ = unclamped.helper(f, f, c, c, 0.0)
    assert (stable_loss >= regular_loss).all()

    torch.manual_seed(0)
    point_f1 = torch.randn(1, 3, 2, 3)
    point_f2 = torch.randn(1, 3, 2, 3)
    point_c = torch.ones(1, 3, 2, 3)
    pointwise = modules.ContrastiveCorrelationLoss(_corr_cfg(pointwise=True))
    pointwise_loss, pointwise_cd = pointwise.helper(point_f1, point_f2, point_c, point_c, 0.0)
    non_pointwise_loss, non_pointwise_cd = unclamped.helper(point_f1, point_f2, point_c, point_c, 0.0)
    assert pointwise_cd.shape == non_pointwise_cd.shape
    assert not torch.equal(pointwise_loss, non_pointwise_loss)
    assert torch.allclose(pointwise_loss.mean(), non_pointwise_loss.mean(), atol=1e-6)


def test_contrastive_correlation_forward_returns_expected_tuple_and_shapes():
    torch.manual_seed(0)
    cfg = _corr_cfg()
    loss = modules.ContrastiveCorrelationLoss(cfg)
    feats = torch.randn(3, 2, 8, 8)
    code = torch.randn(3, 2, 8, 8)
    negative = torch.randn(2, 2, 8, 8)
    result = loss(feats, feats, None, None, code, code, negative, negative)
    assert len(result) == 6
    assert result[0].ndim == 0 and result[2].ndim == 0
    assert result[4].shape[0] == 4

    smaller = loss(feats, feats, None, None, code, code, negative[:2], negative[:2])
    assert smaller[4].shape[0] == min(cfg.neg_samples, min(3, 2)) * min(3, 2)


def test_contrastive_correlation_forward_supports_salience():
    torch.manual_seed(0)
    cfg = _corr_cfg(use_salience=True)
    loss = modules.ContrastiveCorrelationLoss(cfg)
    feats = torch.randn(2, 2, 8, 8)
    code = torch.randn(2, 2, 8, 8)
    salience = torch.zeros(2, 8, 8)
    salience[:, :4, :4] = 1
    result = loss(feats, feats, salience, salience, code, code, feats, code)
    assert len(result) == 6
    assert result[0].ndim == result[2].ndim == 0
    assert result[1].shape == result[3].shape
    assert result[5].shape[1:] == result[1].shape[1:]


def test_contrastive_crf_loss_shape_and_dimension_assertions():
    torch.manual_seed(0)
    loss = modules.ContrastiveCRFLoss(4, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    guidance = torch.randn(2, 3, 5, 5)
    clusters = torch.randn(2, 4, 5, 5)
    assert loss(guidance, clusters).shape == (2, 4, 4)
    with pytest.raises(AssertionError):
        loss(guidance[:1], clusters)
    with pytest.raises(AssertionError):
        loss(guidance, clusters[:, :, :4])


def test_feature_pyramid_helper_resizes_to_56_and_adds_last_dimension():
    x = torch.randn(2, 3, 7, 7)
    actual = modules.FeaturePyramidNet._helper(x)
    assert actual.shape == (2, 3, 56, 56, 1)
