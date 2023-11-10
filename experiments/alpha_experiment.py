import gc
import os

import fire
import numpy as np
import torch


def main(
    temperature: int = 100,
    dstores: str = "both",
    min_alpha: float = 0.5,
    max_alpha: float = 2.0,
    step: float = 0.5,
    model_name: str = "gpt2-large",
    rate_limit: int = 20,
    toxic_dstore: str = "checkpoints/gpt2-large/gpt2_toxicity_gte0.5_clean",
    nontoxic_dstore: str = "checkpoints/gpt2-large/gpt2_toxicity_eq0_half_clean",
):
    """Run prompted generation experiment with varying alpha values.

    Compares the impacts of the ensemble equation alpha parameter on the models'
    toxicity mitigation performance on a subset of 100 prompts.

    Vary alpha from [`min_alpha`, `max_alpha`] by `step` size.

    Args:
        temperature (int, optional): kNN logprobs temperature parameter.
            Defaults to 100.
        dstores (str, optional): Whether to use only 'toxic' or 'both' datastores.
            Defaults to "both".
        min_alpha (float, optional): Minimum value for the ensemble equation
            alpha parameter. Defaults to 0.5.
        max_alpha (float, optional): Maximum value for the ensemble equation
            alpha parameter. Defaults to 2.0.
        step (float, optional): Stepsize to vary from `min_alpha` to `max_alpha`.
            Defaults to 0.5.
        model_name (str, optional): Base model to use. Defaults to "gpt2-large".
        rate_limit (int, optional): Perspective API rate limit. Defaults to 20.
        toxic_dstore (str, optional): Path to the toxic datastore.
        nontoxic_dstore (str, optional): Path to the nontoxic datastore.

    """
    dstore_dirs = ""
    if dstores in ["toxic", "both"]:
        dstore_dirs += f"--dstore_dir {toxic_dstore}"
    if dstores == "both":
        dstore_dirs += f" --other_dstore_dir {nontoxic_dstore}"

    cmd = """
        python -m scripts.run_all \
            --model_name {model_name} \
            --prompts_path data/dexperts/prompts/nontoxic_prompts-10k.jsonl \
            --output_folder outputs/experiments/alpha_temperature/temperature_{temperature}/lambda/{dstores}/lambda_{value} \
            {dstore_dirs} \
            --knn True \
            --method ensemble \
            --lmbda {value} \
            --knn_temp {temperature} \
            --num_prompts 100 \
            --perspective_rate_limit {rate_limit} \
            --batch_size 4
    """

    for value in np.arange(min_alpha, max_alpha + step, step):
        print("\nRunning with lambda = {}".format(value))
        filled_cmd = cmd.format(
            model_name=model_name,
            value=value,
            temperature=temperature,
            dstore_dirs=dstore_dirs,
            dstores=dstores,
            rate_limit=rate_limit,
        )
        os.system(filled_cmd)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    fire.Fire(main)
