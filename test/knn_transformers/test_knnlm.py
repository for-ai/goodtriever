import numpy as np
import pytest
import torch

from generation.knn_transformers.knnlm import KNNWrapper


@pytest.fixture
def lm_logits():
    return torch.FloatTensor([[0.33, 0.33, 0.33]]).log()


@pytest.fixture
def toxic_logits():
    return torch.FloatTensor([[0.0, 0.55, 0.45]]).log()


@pytest.fixture
def nontoxic_logits():
    return torch.FloatTensor([[0.55, 0.0, 0.45]]).log()


def test_interpolate(lm_logits, toxic_logits, lmbda=0.25):
    probs_sanity = KNNWrapper.interpolate(lm_logits, lm_logits, lmbda).exp().squeeze(0)
    probs_toxic = (
        KNNWrapper.interpolate(lm_logits, toxic_logits, lmbda).exp().squeeze(0)
    )

    assert np.isclose(probs_sanity.sum().round(decimals=1), 1), "Probs should sum to 1."
    assert np.isclose(probs_toxic.sum().round(decimals=1), 1), "Probs should sum to 1."

    lm_probs = lm_logits.exp().squeeze(0)
    assert np.allclose(
        probs_sanity.round(decimals=2), lm_probs
    ), "Sending only lm_logits should yield the same result."

    assert (
        probs_toxic[1] > probs_toxic[2] > probs_toxic[0]
    ), "Toxic should be more likely than non-toxic."


def test_interpolate_discourage(lm_logits, toxic_logits, lmbda=0.25):
    probs_sanity = (
        KNNWrapper.interpolate_discourage(lm_logits, lm_logits, lmbda).exp().squeeze(0)
    )
    probs_toxic = (
        KNNWrapper.interpolate_discourage(lm_logits, toxic_logits, lmbda)
        .exp()
        .squeeze(0)
    )

    assert np.isclose(probs_sanity.sum().round(decimals=1), 1), "Probs should sum to 1."
    assert np.isclose(probs_toxic.sum().round(decimals=1), 1), "Probs should sum to 1."

    lm_probs = lm_logits.exp().squeeze(0)
    assert np.allclose(
        probs_sanity.round(decimals=2), lm_probs
    ), "Sending only lm_logits should yield the same result."

    assert (
        probs_toxic[1] < probs_toxic[2] < probs_toxic[0]
    ), "Toxic should be less likely than non-toxic."


def test_ensemble(lm_logits, toxic_logits, nontoxic_logits, lmbda=2.0):
    """Test ensemble equation for different cases."""
    ## Sanity
    probs_sanity = (
        KNNWrapper.ensemble(lm_logits, *(lm_logits, lm_logits), lmbda=lmbda)
        .exp()
        .squeeze(0)
    )

    ## Only toxic
    probs_toxic_ds = (
        KNNWrapper.ensemble(lm_logits, *(toxic_logits, lm_logits), lmbda=lmbda)
        .exp()
        .squeeze(0)
    )
    # Only toxic but not sending lm_logits
    probs_toxic_ds_2 = (
        KNNWrapper.ensemble(lm_logits, *(toxic_logits,), lmbda=lmbda).exp().squeeze(0)
    )

    ## Only non-toxic
    # This is currently not possible in the code
    probs_nontoxic_ds = (
        KNNWrapper.ensemble(lm_logits, *(lm_logits, nontoxic_logits), lmbda=lmbda)
        .exp()
        .squeeze(0)
    )

    ## Both
    probs_both_ds = (
        KNNWrapper.ensemble(lm_logits, *(toxic_logits, nontoxic_logits), lmbda=lmbda)
        .exp()
        .squeeze(0)
    )

    ## Sanity Checks
    lm_probs = lm_logits.exp().squeeze(0)
    assert np.isclose(probs_sanity.sum(), 1), "Probs should sum to 1."
    assert np.isclose(probs_toxic_ds.sum(), 1), "Probs should sum to 1."
    assert np.isclose(probs_nontoxic_ds.sum(), 1), "Probs should sum to 1."
    assert np.isclose(probs_both_ds.sum(), 1), "Probs should sum to 1."

    assert np.allclose(
        probs_sanity.round(decimals=2), lm_probs
    ), "Sending only lm_logits should yield the same result."
    assert np.allclose(
        probs_toxic_ds, probs_toxic_ds_2
    ), "Sending or not sending lm_logits as second element for ensemble should yield the same result."

    ## Token by token comparisons
    # 1st token -> non toxic token
    idx = 0
    assert (
        probs_toxic_ds[idx] > lm_probs[idx]
    ), "Non-toxic token should have higher prob after ensemble with toxic ds."
    assert (
        probs_nontoxic_ds[idx] > lm_probs[idx]
    ), "Non-toxic token should have higher prob after ensemble with nontoxic ds."
    assert (
        probs_both_ds[idx] > lm_probs[idx]
    ), "Non-toxic token should have higher prob after ensemble with both ds."

    # 2nd token -> toxic token
    idx = 1
    assert (
        probs_toxic_ds[idx] < lm_probs[idx]
    ), "Toxic token should have lower prob after ensemble with toxic ds."
    assert (
        probs_nontoxic_ds[idx] < lm_probs[idx]
    ), "Toxic token should have lower prob after ensemble with nontoxic ds."
    assert (
        probs_both_ds[idx] < lm_probs[idx]
    ), "Toxic token should have lower prob after ensemble with both ds."

    # 3rd token -> both toxic and nontoxic token
    idx = 2
    assert (
        probs_toxic_ds[idx] > probs_toxic_ds[1]
    ), "Mixed token should higher prob than toxic token after ensemble with toxic ds."
    assert (
        probs_nontoxic_ds[0] > probs_nontoxic_ds[idx]
    ), "Non-toxic token should higher prob than mixed token after ensemble with nontoxic ds."
    assert (
        probs_both_ds[idx] > probs_both_ds[1]
    ), "Mixed token should higher prob than toxic token after ensemble with both ds."
