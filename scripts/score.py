"""Score sequence of texts with PerspectiveAPI for toxicity.

Heavily inspired by:
https://github.com/allenai/real-toxicity-prompts/blob/master/scripts/run_prompts_experiment.py
"""
from pathlib import Path
from typing import Optional, List

import fire
import numpy as np
import pandas as pd

from utils.constants import PERSPECTIVE_API_KEY
from utils.perspective_api import PerspectiveWorker
from utils.utils import load_cache, structure_output_filepath

if PERSPECTIVE_API_KEY is None:
    raise ValueError(
        "Please run `export PERSPECTIVE_API_KEY=´key´ if you wish to use PerspectiveAPI."
    )


def main(
    input_filename: str,
    column_name: str = "generations",
    output_folder: Optional[str] = None,
    perspective_rate_limit: int = 1,
) -> None:
    """Score sequences of text with PerspectiveAPI.

    Args:
        input_filename (str, optional): jsonl file with generated text to be scored.
            Should be stored locally.
        column_name (str, optional): Name of the field where the text sequences are.
            Supports any dict or list of dict column as long as the dict contains a
            `text` keyword.
            Defaults to "generations".
        output_folder (str, optional): Output folder. If None, results will be saved
            to the same folder as `input_filename`. Defaults to None.
        perspective_rate_limit (int, optional): Maximum number of API calls per second.
            Defaults to 1.

    Raises:
        NotImplementedError: If `column_name` values are not lists or dicts or don't
            have a 'text' key.
    """
    input_filename = Path(input_filename)
    if not input_filename.exists():
        raise ValueError(f"{input_filename} not found.")

    output_file = structure_output_filepath(
        step="perspective",
        output_folder=output_folder or input_filename.parent,
        previous_filename=input_filename,
    )

    df = pd.read_json(input_filename, lines=True)

    if isinstance(df.iloc[0][column_name], dict):
        df[column_name] = df[column_name].apply(lambda x: [x.get("text")])
    elif isinstance(df.iloc[0][column_name], list):
        df[column_name] = df[column_name].apply(
            lambda y: [x.get("text") if isinstance(x, dict) else x for x in y]
        )
    elif isinstance(df.iloc[0][column_name], str):
        df[column_name] = df[column_name].apply(lambda x: [x])
    else:
        raise NotImplementedError(
            "If dict or list of dicts, make sure there's a `text` key."
        )

    num_samples = len(df.iloc[0][column_name])
    perspective = PerspectiveWorker(
        out_file=output_file,
        total=df.shape[0] * num_samples,
        rate_limit=perspective_rate_limit,
    )

    # Flatten and make list
    values = np.stack(df[column_name].values).reshape(-1).tolist()
    del df

    num_cached_scores = load_cache(output_file)
    values = values[num_cached_scores:]

    if len(values) == 0:
        print("No more samples to score.")
        perspective.stop()
        return output_file

    for i, text in enumerate(values):
        perspective(f"generation-{num_cached_scores + i}", text)

    perspective.stop()

    return output_file


if __name__ == "__main__":
    fire.Fire(main)
