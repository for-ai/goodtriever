import tempfile
from pathlib import Path

import datasets
import pandas as pd

from scripts.evaluate import compute_ppl


def test_perplexity():
    """Test if perplexity outputs comparable scores to GPT2 paper.

    In the original paper, wikitext is detokenized. Here we're using the
    raw dataset as pointed out in this reddit discussion by the authors:

    https://www.reddit.com/r/MachineLearning/comments/oye64h/r_struggling_to_reproduce_perplexity_benchmarks/

    """
    df = pd.DataFrame(
        {
            "text": datasets.load_dataset(
                "wikitext", "wikitext-2-raw-v1", split="test"
            )["text"]
        }
    )
    tf = tempfile.NamedTemporaryFile()
    ppl = compute_ppl(
        df.iloc[:100], model_name="gpt2", output_file=Path(tf.name), prompted=False
    )
    results = pd.read_csv(tf.name, index_col=0)

    assert ppl.unprompted.perplexity <= 30
    assert ppl.round(2).equals(results.round(2))
