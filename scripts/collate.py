"""Collate generated text with its toxicity score into a jsonl file.

Heavily inspired by:
https://github.com/allenai/real-toxicity-prompts/blob/master/scripts/run_prompts_experiment.py
"""
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.perspective_api import unpack_scores
from utils.utils import structure_output_filepath


def make_generations_col(generations: List[str], responses: List[Dict]) -> Dict:
    """Join generation text to its results from PerpectiveAPI scores into a dict."""
    for generation, response in zip(generations, responses):
        if isinstance(response, dict):
            response = unpack_scores(response)[0]
        else:
            response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        yield {"text": generation, **response}


def collate(
    generations: List[str],
    responses: Iterable[Dict[str, Any]],
    prompts: Optional[pd.DataFrame],
    prompt_indexes: Optional[List[int]],
) -> pd.DataFrame:
    """Collate generations texts with their scores by perspective API."""
    generations_col_iter = make_generations_col(generations, responses)
    generations_col = list(
        tqdm(
            generations_col_iter,
            total=len(generations),
            desc="Collating files",
            position=1,
        )
    )
    dataset = pd.DataFrame(generations_col)

    # Annotate to which prompt each generation belongs to then groupby
    if prompts is not None and prompt_indexes is not None:
        prompts = prompts.iloc[np.unique(prompt_indexes)]
        dataset["prompt"] = prompt_indexes
        dataset = (
            dataset.groupby("prompt")
            .apply(lambda x: x.to_dict(orient="records"))
            .to_frame()
            .rename(columns={0: "generations"})
        )
        if "generations" in prompts:
            del prompts["generations"]
        dataset = pd.merge(prompts, dataset, left_index=True, right_index=True)

    return dataset


def main(
    generations_path: Union[str, Path],
    scores_path: Union[str, Path],
    prompts_path: str,
    output_folder: Union[str, Path] = None,
    chunksize: int = int(1e5),
    column_name: str = "generations",
) -> Union[str, Path]:
    """Collate generations with its PerspectiveAPI toxicity scores and pre-scored prompts.

    `prompts_path` points to a file that contains a `prompt` column with dict values.
    These dictionaries are pre-scored prompts by PerspectiveAPI and their text.

    Args:
        generations_path (str): Path to generations file.
        scores_path (str): Path to scores file.
        prompts_path (str): Path to prompts file.
        output_folder (str, optional): Output folder. If None, file will be saved to
            `scores_path` folder. Defaults to None.
        chunksize (int): Chunksize to split large scores files when loading
            with pandas. Default value chosen as a reasonable number that usually
            fits memory. Defaults to 100_000.
        column_name (str): Column name where the generations or its text are located.
            Defaults to "generations".
    """
    scores_path = Path(scores_path)
    output_file = structure_output_filepath(
        step="collate",
        output_folder=output_folder or scores_path.parent,
        previous_filename=scores_path.name,
    )

    if output_file.exists():
        print(f"Collated file {output_file} already exists. Returning.")
        return output_file

    generations = pd.read_json(generations_path, lines=True)

    if isinstance(generations[column_name].iloc[0], dict):
        gen_list = np.stack(generations[column_name].apply(lambda x: x.get("text", "")))
    elif isinstance(generations[column_name].iloc[0], list):
        gen_list = np.stack(
            generations[column_name].apply(
                lambda xs: [x.get("text", "") for x in xs]
                if isinstance(xs[0], dict)
                else xs
            )
        )
    else:
        gen_list = np.stack(generations[column_name])

    num_gens = 1 if len(gen_list.shape) == 1 else gen_list.shape[1]
    gen_list = gen_list.reshape(-1).tolist()

    if chunksize % num_gens != 0:
        raise ValueError(
            "`chunksize` should be divisible by the number of generations for each prompt "
            f"The number of continuations for each prompt are: {gen_list.shape[1]}."
        )

    # Generate indexes based on original prompts
    prompt_indexes = (
        np.repeat(generations.index.values, num_gens) if num_gens > 1 else None
    )
    prompts = (
        pd.read_json(prompts_path, lines=True) if prompt_indexes is not None else None
    )

    if prompts is not None and not generations.index.equals(prompts.index):
        warnings.warn("Generations and Prompts indexes do not match.")

    scores = pd.read_json(scores_path, lines=True, chunksize=chunksize)
    lines = 0
    for chunk_idx, chunk in enumerate(
        tqdm(scores, desc="Collating chunks", position=0)
    ):
        start = chunksize * chunk_idx
        end = start + chunksize
        indexes = prompt_indexes[start:end] if prompt_indexes is not None else None

        if "response" in chunk:
            scores_list = chunk["response"].tolist()
        elif "attributeScores" in chunk:
            scores_list = chunk.to_dict(orient="records")

        scored_gens = collate(gen_list[start:end], scores_list, prompts, indexes)

        with open(output_file, "a") as f:
            for _, row in scored_gens.iterrows():
                print(json.dumps(row.to_dict()), file=f)
                lines += 1

    if lines != len(generations):
        warnings.warn(
            f"Number of lines of collated data is {lines}, but was expecting {len(generations)}"
        )

    return output_file


if __name__ == "__main__":
    fire.Fire(main)
