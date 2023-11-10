import gc
import logging
import time
import warnings
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch

from experiments.logger import configure_logger
from generation.args import GenerationParser
from scripts.collate import main as collate
from scripts.evaluate import main as evaluate
from scripts.generate import main as generate
from scripts.score import main as score


def main(
    perspective_rate_limit: int = 90,
    perplexity_model: str = "gpt2-xl",
    collate_chunksize: int = int(1e5),
    sample_perplexity: int = 1000,
    group_toxicity_by: Optional[str] = None,
) -> None:
    """Run full pipeline: generate, score, collate and evaluate.

    All generation arguments are supported. For more info check generate.py
    and generation/args.py file.

    Args:
        perspective_rate_limit (int, optional): Maximum number of PerspectiveAPI
            calls per second. Defaults to 110.
        perplexity_model (str, optional): Model to compute perplexity with.
            Defaults to "gpt2-xl".
        collate_chunksize (int, optional): Used in the collate script.
            Chunksize to split large files when loading with pandas.
            Default value chosen as a reasonable number that usually
            fits memory. Defaults to 100_000.
        sample_perplexity (int, optional): Used in the evaluate script.
            Number of prompts to compute perplexity for.
            Defaults to 1000.
        group_toxicity_by (str, optional): Column to group toxicity results by
            (i.e. a column containing different classes of interest). Only
            possible for prompted generation. Classes should be present in the
            `prompts` file. Defaults to None.

    """
    parser = GenerationParser()

    logfile = Path(parser.gen_args.output_folder) / "logs/run_all.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    configure_logger(filename=logfile, level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Generate
    num_instances = 0
    time_taken = []
    start = time.time()
    start_instance = time.time()
    for _ in generate(parser):
        if num_instances > 0:
            time_taken.append(
                (time.time() - start_instance) / parser.gen_args.num_return_sequences
            )
        num_instances += 1
        start_instance = time.time()
        pass
    end = time.time()
    total_sequences = num_instances * parser.gen_args.num_return_sequences
    logger.info(f"Generation took {end - start:.5f} seconds.")
    logger.info(
        f"Number of generated sequences: {total_sequences} // Num instances: {num_instances}"
    )
    logger.info(
        f"Actual time per sequence of {parser.gen_args.max_new_tokens} tokens: "
        f"{np.mean(time_taken):.5f} ({np.std(time_taken):.5f}) seconds."
    )

    # To avoid faiss error: "Failed to cudaFree pointer"
    torch.cuda.empty_cache()
    gc.collect()

    output_folder = Path(parser.gen_args.output_folder)
    generations_path = str(output_folder / parser.gen_args.output_filename)

    start = time.time()
    scores_path = score(
        input_filename=generations_path,
        output_folder=output_folder,
        perspective_rate_limit=perspective_rate_limit,
    )
    logger.info(f"Scoring took {time.time() - start:.2f} seconds.")

    start = time.time()
    collated_path = collate(
        generations_path=generations_path,
        scores_path=scores_path,
        output_folder=output_folder,
        chunksize=collate_chunksize,
        prompts_path=parser.gen_args.prompts_path,
    )
    logger.info(f"Collating took {time.time() - start:.2f} seconds.")


    if "prompted_" not in collated_path.name:
        warnings.warn(
            f"Invalid filename for {collated_path}. No 'eos_' or 'prompted_' strings found."
        )

    start = time.time()
    evaluate(
        prompted_json=collated_path,
        compute_perplexity=True,
        compute_toxicity=True,
        compute_diversity=True,
        model_name=perplexity_model,
        sample_perplexity=sample_perplexity,
        group_toxicity_by=group_toxicity_by,
    )
    logger.info(f"Evaluation took {time.time() - start:.2f} seconds.")


if __name__ == "__main__":
    fire.Fire(main)
