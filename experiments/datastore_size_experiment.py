import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import fire
import pandas as pd

from experiments.continual_learning.components import build_dstore, train_expert
from experiments.continual_learning.utils import evaluate
from experiments.logger import configure_logger


def main(
    model_name: str,
    prompts_path: str,
    perplexity_model: str = None,
    toxic_tokens: Union[int, Tuple] = None,
    nontoxic_tokens: Union[int, Tuple] = (10_000, 500_000, None),
    num_prompts: Optional[int] = None,
    output_folder: str = "outputs/experiments/",
    experiment_name: str = "datastore_size",
    perspective_rate_limit: int = 30,
    toxic_train_file: str = "data/jigsaw/toxicity_gte0.5_clean.json",
    nontoxic_train_file: str = "data/jigsaw/toxicity_eq0_half_clean.json",
    kind: str = "knn",
    lmbda: float = 2.0,
    knn_temp: int = 100,
    filter_p: float = 0.9,
    group_results_by: Optional[str] = None,
    batch_size: int = 4,
    custom_attrs: Optional[Tuple[str]] = None,
):
    """Run prompted generation experiment with varying datastore sizes.

    Define the size of datastores by setting a list to `toxic_tokens`
    and `nontoxic_tokens`. The list should be of equal size and experiments
    are define by the index of those list (i.e. index 0 defines the amount of
    tokens for that datastore on run 0).

    If any of `toxic` or `nontoxic_tokens` is set to an integer,
    it will be repeated to match the size of the other.

    Args:
        model_name (str, optional): Base model to use.
        prompts_path (str, optional): Path to prompts path.
        perplexity_model (str, optional): Perplexity model to use.
        toxic_tokens (Union[int, Tuple], optional): Tuple of the size of the
            toxic datastore at each experiment. If None, uses all available tokens.
            Defaults to None.
        nontoxic_tokens (Union[int, Tuple], optional): Tuple of the size of the
            toxic datastore at each experiment. If None, uses all available tokens.
            Defaults to (10_000, 500_000, None).
        num_prompts (Optional[int], optional): Number of prompts to run experiments on.
            If None, run to all available prompts. Defaults to None.
        output_folder (str, optional): Name of output folder.
            Defaults to "outputs/experiments/".
        experiment_name (str, optional): Name of experiment. Defaults to "datastore_size".
        perspective_rate_limit (int, optional): Perspective API Rate Limit. Defaults to 30.
        toxic_train_file (str, optional): Path to train file with toxic comments.
            Defaults to "data/jigsaw/toxicity_gte0.5_clean.json".
        nontoxic_train_file (str, optional): Path to train file with nontoxic comments.
            Defaults to "data/jigsaw/toxicity_eq0_half_clean.json".
        kind (str, optional): Kind of experiment to run. Defaults to "knn".
        lmbda (float, optional): Lambda (interpolation) or alpha (ensemble) value.
            Defaults to 2.0.
        knn_temp (int, optional): kNN logprobs temperature parameter.
            Defaults to 100.
        filter_p (float, optional): Filter probability for kNN logprobs.
            Defaults to 0.9.
        group_results_by (Optional[str], optional): Group results by attribute.
            Defaults to None.
        batch_size (int, optional): Batch size for generation. Defaults to 4.
        custom_attrs (Optional[List[str]], optional): List of attributes to be
            evaluated by Perspective API. If None, all attributes will be used. Some
            languages only support "TOXICITY,". For more options, check Perspective's
            documentation. Defaults to None.

    Raises:
        NotImplementedError: Raised when unsupported option of `kind` is given.
        NotImplementedError: Raised when unsupported option of `dstores` is given.

    """
    if kind != "knn":
        raise NotImplementedError(f"Only 'knn' is supported. Got '{kind}'.")

    output_folder = Path(output_folder) / experiment_name / model_name

    if not isinstance(toxic_tokens, Iterable) and isinstance(nontoxic_tokens, Iterable):
        toxic_tokens = (toxic_tokens,) * len(nontoxic_tokens)
    if not isinstance(nontoxic_tokens, Iterable) and isinstance(toxic_tokens, Iterable):
        nontoxic_tokens = (nontoxic_tokens,) * len(toxic_tokens)

    assert len(toxic_tokens) == len(
        nontoxic_tokens
    ), "Must have same number of dstore sizes."

    (output_folder / "logs").mkdir(parents=True, exist_ok=True)
    configure_logger(output_folder / "logs/experiment.log")
    logger = logging.getLogger(__name__)

    for dstore_tokens, other_dstore_tokens in zip(toxic_tokens, nontoxic_tokens):
        curr_folder = (
            output_folder / f"toxic={dstore_tokens}_nontoxic={other_dstore_tokens}"
        )
        (curr_folder / "logs").mkdir(parents=True, exist_ok=True)

        logger.info(f"{'====' * 5}")
        logger.info(
            f"Starting '{experiment_name}/{model_name}/toxic={dstore_tokens}_nontoxic={other_dstore_tokens}' experiment."
        )
        logger.info(f"{'====' * 5}")

        paths = defaultdict(lambda: None)
        for i, train_file in enumerate([toxic_train_file, nontoxic_train_file]):
            toxicity = "toxic" if i == 0 else "nontoxic"
            tokens = dstore_tokens if i == 0 else other_dstore_tokens
            if tokens:
                logger.info(f"Datastore limited to {tokens} tokens.")
            else:
                logger.info(f"Unlimited datastore.")

            # Skipping dstore training logics
            done_domains = (
                open(output_folder / "logs/done_domains.txt").readlines()
                if (output_folder / "logs/done_domains.txt").exists()
                else []
            )
            done = False
            if any(f"{tokens}, {toxicity}" in done for done in done_domains):
                logger.info(f"Skipped {tokens}, {toxicity} build or train")
                done = True

            model_path = output_folder / f"checkpoints/{toxicity}/{tokens}"
            if kind == "knn":
                path = build_dstore(
                    output_folder=model_path,
                    train_file=train_file,
                    model_name=model_name,
                    toxicity=toxicity,
                    dstore_size=tokens,
                    log_folder=curr_folder / f"logs/build_dstore_{toxicity}.log",
                    continue_writing=False,
                    done=done,
                )
            else:
                raise NotImplementedError(f"`kind` = {kind} not supported")
            paths[toxicity] = path

            if not done:
                with open(output_folder / "logs/done_domains.txt", "a") as f:
                    f.write(f"{tokens}, {toxicity}\n")

        evaluate(
            domain=f"dstore_size - {tokens}",
            output_folder=curr_folder,
            model_name=model_name,
            prompts_path=prompts_path,
            toxic_model=paths["toxic"],
            nontoxic_model=paths["nontoxic"],
            rate_limit=perspective_rate_limit,
            group_results_by=group_results_by,
            num_prompts=num_prompts,
            kind=kind,
            batch_size=batch_size,
            perplexity_model=perplexity_model,
            custom_attrs=custom_attrs,
            lmbda=lmbda,
            knn_temp=knn_temp,
            filter_p=filter_p,
        )


if __name__ == "__main__":
    fire.Fire(main)
