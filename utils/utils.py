"""Utility functions.

Some of the functions are from
https://github.com/allenai/real-toxicity-prompts/blob/master/utils/utils.py
"""
import json
from pathlib import Path
from typing import Iterable, List, Optional, TypeVar, Union

from tqdm.auto import tqdm

T = TypeVar("T")


def structure_output_filepath(
    step: str,
    previous_filename: Union[Path, str],
    output_folder: Optional[Path] = None,
    mkdir: bool = True,
):
    """Structure output filename given a step, output folder and previous filename."""
    if isinstance(previous_filename, str):
        previous_filename = Path(previous_filename)

    stem = previous_filename.stem

    if output_folder is None:
        output_folder = previous_filename.parent

    if isinstance(output_folder, str):
        output_folder = Path(output_folder)

    if step == "generation":
        output_file = f"{stem}_generations.jsonl"
    elif step == "perspective":
        if "generations" in stem:
            output_file = f"{stem.replace('generations', 'perspective')}.jsonl"
        else:
            output_file = f"{stem}_perspective.jsonl"
    elif step == "collate":
        if "perspective" in stem in stem:
            output_file = f"{stem.replace('perspective', 'collated')}.jsonl"
        else:
            output_file = f"{stem}_collated.jsonl"
    elif step == "toxicity":
        if "collated" in stem in stem:
            output_file = f"{stem.replace('collated', 'toxicity')}.csv"
        else:
            output_file = f"{stem}_toxicity.csv"
    elif step == "perplexity":
        if "collated" in stem in stem:
            output_file = f"{stem.replace('collated', 'perplexity')}.csv"
        else:
            output_file = f"{stem}_perplexity.csv"
    elif step == "diversity":
        if "collated" in stem in stem:
            output_file = f"{stem.replace('collated', 'diversity')}.csv"
        else:
            output_file = f"{stem}_diversity.csv"
    else:
        raise NotImplementedError(
            f"Step {step} not implemented for automatic filename structuring."
        )

    output_file = output_folder / output_file

    if mkdir:
        output_file.parent.mkdir(exist_ok=True, parents=True)

    print(f"Saving to {output_file}.")
    return output_file


def _load_cache(file: Path):
    """Load json file and return number of cached lines."""
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f"Loading cache from {file}"):
                yield json.loads(line)


def load_cache(file: Path) -> int:
    """Load json file and return number of cached lines."""
    lines = 0
    for _ in _load_cache(file):
        lines += 1

    return lines


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    """Create batches of `batch_size` from an iterable."""
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch
