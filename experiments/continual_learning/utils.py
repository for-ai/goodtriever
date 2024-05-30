import gc
import logging
import re
import subprocess
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def cleanup(func):
    """Empty cache from cuda and system."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start_time}")
        torch.cuda.empty_cache()
        gc.collect()

    return wrapper


def run(cmd):
    """Run a subprocess and clean things up when it's finished"""
    return cleanup(subprocess.run(cmd, shell=True))


def evaluate(
    domain,
    output_folder,
    model_name,
    prompts_path,
    toxic_model,
    nontoxic_model,
    rate_limit,
    group_results_by=None,
    num_prompts=None,
    kind="dexperts",
    batch_size=4,
    flat_index=False,
    perplexity_model="gpt2-xl",
    custom_attrs=None,
    lmbda=2.0,
    knn_temp=100,
    filter_p=0.9,
):
    """Run `run_all` script. Generate completions, score and evaluate."""

    generate_cmd = f"""
        python -m scripts.run_all \
            --output_folder {output_folder} \
            --model_name {model_name} \
            --prompts_path {prompts_path} \
            --dstore_dir {toxic_model} \
            {f'--other_dstore_dir {nontoxic_model}' if nontoxic_model else ''} \
            {'--dexperts True' if kind == 'dexperts' else ''} \
            {'--knn True' if kind == 'knn' else ''} \
            {'--flat_index True' if flat_index else ''} \
            {f'--num_prompts {num_prompts}' if num_prompts is not None else ''} \
            {f'--custom_attrs {",".join(custom_attrs)},' if custom_attrs is not None else ''} \
            --perspective_rate_limit {rate_limit} \
            --perplexity_model {perplexity_model} \
            --method ensemble \
            --lmbda {lmbda} \
            --knn_temp {knn_temp} \
            --filter_p {filter_p} \
            --group_results_by {group_results_by} \
            --batch_size {batch_size}
    """
    logger.info(f"Running domain {domain} `run_all` command: {generate_cmd}")
    return run(generate_cmd)


def extract_domains_from_file_list(file_list):
    """Extract domains from files found within a filename pattern."""

    def _extract_domain(text):
        pattern = r"(.*?)_(toxic|nontoxic)\.json"
        matches = re.findall(pattern, text)

        if matches:
            for match in matches:
                return match[0]

    return sorted(set([_extract_domain(str(filename)) for filename in file_list]))


def setup_output_folder(
    output_folder, toxicity_choices, experiment_name, model_name, kind
):
    if not all(choice in ["toxic", "nontoxic"] for choice in toxicity_choices):
        raise ValueError(
            "`toxicity_choices` should contain only 'toxic' or 'nontoxic'."
        )

    # Toxicity and dstores setup
    toxic_added = True if "toxic" in toxicity_choices else False
    nontoxic_added = True if "nontoxic" in toxicity_choices else False

    # Folder setup
    output_folder = (
        Path(output_folder)
        / experiment_name
        / model_name
        / kind
        / f"toxic={toxic_added}_nontoxic={nontoxic_added}"
    )

    return output_folder
