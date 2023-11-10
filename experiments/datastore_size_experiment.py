import gc
import logging
import os
import time
from pathlib import Path
from typing import Tuple, Optional, Union, Iterable

import fire
import torch
from logger import configure_logger


def main(
    toxic_tokens: Union[int, Tuple] = None,
    nontoxic_tokens: Union[int, Tuple] = (10_000, 500_000, None),
    num_prompts: Optional[int] = None,
    output_folder: str = "outputs/experiments/",
    experiment_name: str = "datastore_size",
    rate_limit: int = 30,
    dstores: str = "both",
    model_name: str = "gpt2-large",
    toxic_train_file: str = "data/jigsaw/toxicity_gte0.5_clean.json",
    nontoxic_train_file: str = "data/jigsaw/toxicity_eq0_half_clean.json",
    method: str = "ensemble",
    lmbda: float = 2.0,
    temperature: int = 100,
    prompts_path: str = "data/dexperts/prompts/nontoxic_prompts-10k.jsonl",
    only_generate: bool = False,
):
    """Run prompted generation experiment with varying datastore sizes.

    Define the size of datastores by setting a list to `toxic_tokens`
    and `nontoxic_tokens`. The list should be of equal size and experiments
    are define by the index of those list (i.e. index 0 defines the amount of
    tokens for that datastore on run 0).

    If any of `toxic` or `nontoxic_tokens` is set to an integer,
    it will be repeated to match the size of the other.

    Args:
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
        rate_limit (int, optional): Perspective API Rate Limit. Defaults to 30.
        dstores (str, optional): Whether to use only 'toxic' or 'both' datastores.
            Defaults to "both".
        model_name (str, optional): Base model to use. Defaults to "gpt2-large".
        toxic_train_file (str, optional): Path to train file with toxic comments.
            Defaults to "data/jigsaw/toxicity_gte0.5_clean.json".
        nontoxic_train_file (str, optional): Path to train file with nontoxic comments.
            Defaults to "data/jigsaw/toxicity_eq0_half_clean.json".
        method (str, optional): Which method to use. Choices are: ensemble,
            interpolation and interpolation_discourage. Defaults to "ensemble".
        lmbda (float, optional): Lambda (interpolation) or alpha (ensemble) value.
            Defaults to 2.0.
        temperature (int, optional): kNN logprobs temperature parameter.
            Defaults to 100.
        prompts_path (str, optional): Path to prompts path.
            Defaults to "data/dexperts/prompts/nontoxic_prompts-10k.jsonl".
        only_generate (bool, optional): Whether to skip datastore building and
            index training or not. Defaults to False.

    Raises:
        NotImplementedError: Raised when unsupported option of `dstores` is given.

    """
    base_folder = Path(output_folder) / experiment_name / model_name

    if not isinstance(toxic_tokens, Iterable) and isinstance(nontoxic_tokens, Iterable):
        toxic_tokens = (toxic_tokens,) * len(nontoxic_tokens)
    if not isinstance(nontoxic_tokens, Iterable) and isinstance(toxic_tokens, Iterable):
        nontoxic_tokens = (nontoxic_tokens,) * len(toxic_tokens)

    assert len(toxic_tokens) == len(
        nontoxic_tokens
    ), "Must have same number of dstore sizes."

    (base_folder / "logs").mkdir(parents=True, exist_ok=True)
    configure_logger(base_folder / "logs/experiment.log")
    logger = logging.getLogger(__name__)

    for dstore_tokens, other_dstore_tokens in zip(toxic_tokens, nontoxic_tokens):
        dstore_dirs = ["--dstore_dir", "--other_dstore_dir"]
        output_folder = (
            base_folder / f"toxic={dstore_tokens}_nontoxic={other_dstore_tokens}"
        )
        (output_folder / "logs").mkdir(parents=True, exist_ok=True)

        logger.info(f"{'====' * 5}")
        logger.info(
            f"Starting '{experiment_name}/{model_name}/toxic={dstore_tokens}_nontoxic={other_dstore_tokens}' experiment."
        )
        logger.info(f"{'====' * 5}")

        for i, train_file in enumerate([toxic_train_file, nontoxic_train_file]):
            if i == 0:
                tokens = dstore_tokens
            elif i == 1:
                tokens = other_dstore_tokens
                if dstores == "toxic":
                    dstore_dirs = dstore_dirs[:1]
                    continue
            elif dstores == "nontoxic":
                raise NotImplementedError(
                    "Currently, just using both datastores or just the toxic one is supported."
                )

            dstore = (
                base_folder / "checkpoints" / f"gpt2_{Path(train_file).stem}_{tokens}"
            )
            dstore_dirs[i] = f"{dstore_dirs[i]} {dstore}"

            if only_generate or dstore.exists():
                logger.info(
                    f"Skipping datastore build and index train for datastore {i}."
                )
                continue

            if tokens:
                logger.info(f"Datastore limited to {tokens} tokens.")
            else:
                logger.info(f"Unlimited datastore.")

            ds_cmd = f"""
                python -u -m generation.knn_transformers.run_clm \
                    --model_name_or_path {model_name} \
                    --train_file {train_file} \
                    --eval_subset train \
                    --output_dir {dstore} \
                    --dstore_dir {dstore} \
                    {f'--dstore_size {tokens} --limit_eval_to_dstore' if tokens else ''} \
                    --save_knnlm_dstore \
                    --do_eval | tee {output_folder / f"logs/build_dstore_{i}.log"}
            """

            train_cmd = f"""
                python -u -m generation.knn_transformers.run_clm \
                    --model_name_or_path {model_name} \
                    --train_file {train_file} \
                    --eval_subset train \
                    --output_dir {dstore} \
                    --dstore_dir {dstore} \
                    {f'--dstore_size {tokens}' if tokens else ''} \
                    --build_index | tee {output_folder / f"logs/build_index_{i}.log"}
            """
            logger.info(f"Running `datastore build` command {i}: {ds_cmd}")
            start_time = time.time()
            os.system(ds_cmd)
            logger.info(f"Execution time: {time.time() - start_time}")
            torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"Running `index train` command {i}: {train_cmd}")
            start_time = time.time()
            os.system(train_cmd)
            logger.info(f"Execution time: {time.time() - start_time}")
            torch.cuda.empty_cache()
            gc.collect()

        generate_cmd = f"""
            python -m scripts.run_all \
                --output_folder {output_folder} \
                --perspective_rate_limit {rate_limit} \
                --model_name {model_name} \
                --prompts_path {prompts_path} \
                {" ".join(dstore_dirs)} \
                --knn True \
                --method ensemble \
                --lmbda {lmbda} \
                --knn_temp {temperature} \
                --method {method} \
                {f'--num_prompts {num_prompts}' if num_prompts else ''} \
                --batch_size 4
        """

        logger.info(f"Running `run_all` command: {generate_cmd}")
        start_time = time.time()
        os.system(generate_cmd)
        logger.info(f"Execution time: {time.time() - start_time}")
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    fire.Fire(main)
