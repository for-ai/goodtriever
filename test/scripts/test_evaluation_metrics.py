import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.evaluation.evaluation_metrics import compute_distinctness, compute_ppl


@pytest.fixture
def df_with_group():
    df = pd.DataFrame(
        [
            {
                "prompt": {"text": "This is a test sentence.", "toxicity": 0},
                "generations": [
                    {"text": "A third test sentence."},
                ],
                "group": "A",
            },
            {
                "prompt": {"text": "This is a test sentence.", "toxicity": 0},
                "generations": [
                    {"text": "This This This This"},
                ],
                "group": "B",
            },
            {
                "prompt": {"text": "This is a test sentence.", "toxicity": 0},
                "generations": [
                    {"text": "This is a test sentence."},
                ],
                "group": "A",
            },
        ]
    )
    return df


def test_compute_ppl_group_by(df_with_group):
    """Test if compute_ppl function groups results by the specified column."""
    tf = tempfile.NamedTemporaryFile(suffix=".csv")
    if os.path.exists(tf.name):
        os.remove(tf.name)
    ppl = compute_ppl(
        df_with_group,
        model_name="gpt2",
        output_file=Path(tf.name),
        group_results_by="group",
    )
    main_ppl = pd.read_csv(tf.name, index_col=0)
    grouped_ppl_file = Path(tf.name.replace(".csv", "_groupby.csv"))
    grouped_ppl = pd.read_csv(grouped_ppl_file, index_col=0)

    assert set(grouped_ppl.index) == set(["A", "B"])
    assert grouped_ppl.loc["A"]["nontoxic"] > 0
    assert main_ppl["nontoxic"].perplexity > 0


def test_diversity_groupby(df_with_group):
    """Test if diversity function groups results by the specified column."""
    tf = tempfile.NamedTemporaryFile(suffix=".csv")
    grouped_diversity_file = Path(tf.name.replace(".csv", "_groupby.csv"))
    if Path(tf.name).exists():
        os.remove(tf.name)
    if grouped_diversity_file.exists():
        os.remove(tf.name.replace(".csv", "_groupby.csv"))

    diversity = compute_distinctness(
        df_with_group, output_file=Path(tf.name), group_results_by="group"
    )

    main_diversity = pd.read_csv(tf.name, index_col=0)
    grouped_diversity = pd.read_csv(grouped_diversity_file, index_col=0)

    assert grouped_diversity.loc["A"]["dist1"] == 1  # Not a single token repeated
    assert grouped_diversity.loc["B"]["dist1"] == 0.25  # 1 token repeated 4 times
    assert main_diversity["dist1"][0] != 1
