import warnings
from pathlib import Path
from typing import Optional

import fire
import pandas as pd

from scripts.utils.evaluation_metrics import (
    compute_distinctness,
    compute_ppl,
    compute_toxicity_prompted,
    compute_toxicity_unprompted,
)
from utils.utils import structure_output_filepath


def main(
    prompted_json: str,
    compute_perplexity: bool = True,
    compute_toxicity: bool = True,
    compute_diversity: bool = True,
    model_name: str = "gpt2-xl",
    sample_perplexity: Optional[int] = 1000,
    threshold: float = 0.5,
    group_toxicity_by: Optional[str] = None,
):
    """Compute toxicity and perplexity metrics for prompted or unprompted generations.

    The inputs files for this script are the ones from the "collate.py" script.

    Args:
        prompted_json (str): Path to prompted generations jsonl file.
        compute_perplexity (bool, optional): Whether to compute perplexity or not.
            Defaults to True.
        compute_toxicity (bool, optional): Whether to compute toxicity metrics or not.
            Defaults to True.
        compute_diversity (bool, optional): Whether to compute diversity metrics or not.
            Defaults to True.
        model_name (str, optional): Which model to compute perplexity with.
            Defaults to "gpt2-xl".
        sample_perplexity (int, optional): The amount of prompt samples to
            from each toxicity condition to compute perplexity.
            If None, computes for all samples.
            Defaults to None.
        threshold (float, optional): Toxicity threshold.
            If higher than `threshold`, the text is toxic. Defaults to 0.5.
        group_toxicity_by (str, optional): Column to group toxicity results by
            (i.e. a column containing different classes of interest). Only
            possible for prompted generation. Classes should be present in the
            `prompted_json` file. Defaults to None.

    """
    path = Path(prompted_json)
    df = pd.read_json(path, lines=True)

    if compute_toxicity:
        output_file = structure_output_filepath(
            step="toxicity",
            previous_filename=path,
        )
        if not output_file.exists():
            compute_toxicity_prompted(
                df,
                output_file,
                threshold=threshold,
                group_toxicity_by=group_toxicity_by,
            )
        else:
            warnings.warn(f"{output_file} already exists. Skipping.")

    if compute_perplexity:
        output_file = structure_output_filepath(
            step="perplexity",
            previous_filename=path,
        )
        compute_ppl(
            df,
            model_name,
            output_file,
            sample_perplexity=sample_perplexity,
            threshold=threshold,
        )

    if compute_diversity:
        output_file = structure_output_filepath(
            step="diversity",
            previous_filename=path,
        )
        compute_distinctness(df, output_file)


if __name__ == "__main__":
    fire.Fire(main)
