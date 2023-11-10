import gc
import logging
import os
import time
from pathlib import Path

import fire
import torch
from logger import configure_logger


def main(
    model_name: str = "gpt2-large",
    output_folder: str = "outputs/experiments/",
    experiment_name: str = "model_size",
    dstores: str = "both",
    toxic_train_file: str = "data/jigsaw/toxicity_gte0.5_clean.json",
    nontoxic_train_file: str = "data/jigsaw/toxicity_eq0_half_clean.json",
    method: str = "ensemble",
    lmbda: float = 2.0,
    temperature: int = 100,
    prompts_path: str = "data/dexperts/prompts/nontoxic_prompts-10k.jsonl",
    only_generate: bool = False,
):
    """Run full prompted generation experiment for a given HuggingFace model.

    Usually performed for different sizes of a same model family, i.e.
    gpt2, gpt2-medium and gpt-large.

    Args:
        model_name (str, optional): Base model to use. Defaults to "gpt2-large".
        output_folder (str, optional): Name of output folder.
            Defaults to "outputs/experiments/".
        experiment_name (str, optional): Name of experiment. Defaults to "model_size".
        dstores (str, optional): Whether to use only 'toxic' or 'both' datastores.
            Defaults to "both".
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
    output_folder = Path(output_folder) / experiment_name / model_name
    (output_folder / "logs").mkdir(parents=True, exist_ok=True)

    configure_logger(output_folder / "logs/experiment.log")
    logger = logging.getLogger(__name__)

    logger.info(f"Starting '{experiment_name}/{model_name}' experiment.")

    dstore_dirs = ["--dstore_dir", "--other_dstore_dir"]
    for i, train_file in enumerate([toxic_train_file, nontoxic_train_file]):
        if dstores == "toxic":
            if i == 1:
                dstore_dirs = dstore_dirs[:1]
                continue
        elif dstores == "nontoxic":
            raise NotImplementedError(
                "Currently, just using both datastores or just the toxic one is supported."
            )

        dstore = output_folder / f"checkpoints/gpt2_{Path(train_file).stem}"
        dstore_dirs[i] = f"{dstore_dirs[i]} {dstore}"

        if only_generate:
            continue

        ds_cmd = f"""
            python -u -m generation.knn_transformers.run_clm \
                --model_name_or_path {model_name} \
                --train_file {train_file} \
                --eval_subset train \
                --output_dir {dstore} \
                --dstore_dir {dstore} \
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
            --model_name {model_name} \
            --prompts_path {prompts_path} \
            {" ".join(dstore_dirs)} \
            --knn True \
            --method ensemble \
            --lmbda {lmbda} \
            --knn_temp {temperature} \
            --method {method} \
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
