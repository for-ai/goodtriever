"""Generate continuations for a given prompt and model_name.

Heavily inspired by:
https://github.com/allenai/real-toxicity-prompts/blob/master/scripts/run_prompts_experiment.py
"""
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from generation.args import GenerationParser
from generation.base import batched_generation
from generation.models import setup_model, setup_tokenizer
from utils.utils import load_cache, structure_output_filepath

ALLOWED_MODELS = ["gpt2", "gpt2-medium", "gpt2-large"]


def build_filename(gen_args, knn_args) -> str:
    """Build filename from user arguments."""
    name = f"prompted_{gen_args.model_name}"

    if knn_args.knn:
        name += "_knn"
        name += f"_{str(knn_args.lmbda).replace('.','')}"
        name += f"_{knn_args.method}"
    return name


def main(parser: Optional = None) -> Iterable:
    """Generate sequences of text with HuggingFace models.

    By default, the kNN retrieval system is deactivated and generations
    are performed with the base model as defined by `model_name`.
    To check which arguments are available, type `python -m scripts.generate -h`

    Further instructions can be found at the `knn_transformers` folder README.

    Yields:
        np.array: Generated sequence array.
    """

    if parser is None:
        parser = GenerationParser()

    gen_args, knn_args = parser.gen_args, parser.knn_args

    df = pd.read_json(gen_args.prompts_path, lines=True)
    if "prompt" in df.columns:
        df = pd.json_normalize(df["prompt"])

    # Create base filename
    if gen_args.output_filename is None:
        gen_args.output_filename = build_filename(gen_args, knn_args)

    output_file = structure_output_filepath(
        step="generation",
        output_folder=Path(gen_args.output_folder),
        previous_filename=gen_args.output_filename,
    )
    # Update name
    gen_args.output_filename = output_file.name

    # Save generation args
    args_filename = output_file.parent / "prompted_args.json"
    with open(args_filename, "w") as f:
        f.write(json.dumps(parser.all_args, indent=2, default=str))

    # Remove prompts that have already been generated
    lines = load_cache(output_file)
    df = df.iloc[lines:]

    if gen_args.num_prompts is not None:
        if lines <= gen_args.num_prompts:
            df = df.iloc[: (gen_args.num_prompts - lines)]

    if df.empty:
        return

    tokenizer = setup_tokenizer(gen_args.model_name)
    model = setup_model(gen_args.model_name, knn_args)

    yield from batched_generation(
        output_file,
        df,
        model,
        tokenizer,
        batch_size=gen_args.batch_size,
        num_return_sequences=gen_args.num_return_sequences,
        max_new_tokens=gen_args.max_new_tokens,
        top_p=gen_args.top_p,
    )


if __name__ == "__main__":
    for _ in main():
        pass
