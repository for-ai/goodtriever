import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import fire
import pandas as pd

from experiments.continual_learning.components import build_dstore, train_expert
from experiments.continual_learning.utils import (
    evaluate,
    extract_domains_from_file_list,
    setup_output_folder,
)
from experiments.logger import configure_logger

logger = logging.getLogger(__name__)


def main(
    model_name: str,
    perplexity_model: str,
    prompts_path: str,
    train_folder: str,
    kind: str = "knn",
    domains: Optional[Tuple] = None,
    toxicity_choices: Tuple = ("toxic", "nontoxic"),
    toxic_pattern: str = "wilds_*_toxic.json",
    nontoxic_pattern: str = "wilds_*_nontoxic.json",
    output_folder: str = "outputs/experiments/continual_learning",
    experiment_name: str = "continual_mitigation",
    perspective_rate_limit: str = 90,
    batch_size: int = 4,
    group_results_by: str = "domain",
    pretrained_toxic: Optional[str] = None,
    pretrained_nontoxic: Optional[str] = None,
    dstore_size: Optional[int] = None,
    num_prompts: Optional[int] = None,
    multitask: Optional[bool] = True,
    custom_attrs: Optional[List[str]] = None,
    lmbda: float = 2.0,
    knn_temp: int = 100,
    filter_p: float = 0.9,
    learning_rate: float = 5e-5,
):
    """Run continual learning experiments.

    If multitask is True (default), data is continually added to Goodtriever's datastores
    and DExperts' train file. If False, only the current domain is used for training or
    in the datastore.

    Args:
        model_name (str, optional): Base model, from huggingface.
        perplexity_model (str, optional): Model for ppl evaluation, from huggingface.
        prompts_path (str, optional): Path to prompts file.
        train_folder (str, optional): Folder that contains all training files.
        kind (str, optional): Options are ('knn', 'dexperts'). Defaults to "knn".
        domains (Optional[Tuple], optional): The list of domains. If None,
            they will be inferred from files. Defaults to None.
        toxicity_choices (Tuple, optional): Which datastores will be continually improved.
            Defaults to ("toxic", "nontoxic").
        toxic_pattern (str, optional): Pattern to find toxic files in `train_folder`.
            Defaults to "wilds_*_toxic.json".
        nontoxic_pattern (str, optional): Pattern to find nontoxic files in `train_folder`.
            Defaults to "wilds_*_nontoxic.json".
        output_folder (str, optional): Folder to save experiments output.
            Defaults to "outputs/experiments/continual_learning".
        experiment_name (str, optional): Experiment name will create a new folder inside `output_folder`.
            Defaults to "continual_mitigation".
        perspective_rate_limit (str, optional): Perspective API rate limit. Defaults to 90.
        batch_size (int, optional): Batch size for inference. Defaults to 4.
        group_results_by (str, optional): This column should be in the prompts file and
            it is used to compute domain-specific toxicity metrics. Defaults to "domain".
        pretrained_toxic (Optional[str], optional): Path to a pretrained
            toxic datastore or expert. Defaults to None.
        pretrained_nontoxic (Optional[str], optional): Path to a pretrained
            nontoxic datastore or expert. Defaults to None.
        dstore_size (Optional[int], optional): Set if you want to limit datastore size.
            Defaults to None.
        num_prompts (Optional[int], optional): Number of prompts to be evaluated.
            If None, will use all. Defaults to None.
        multitask (Optional[bool], optional): If True, DExperts and Goodtriever will be run
            in the continual learning manner. At every step, DExperts will be
            trained with all data available. Goodtriever will update a previously
            updated datastore. Defaults to True.
        custom_attrs (Optional[List[str]], optional): List of attributes to be
            evaluated by Perspective API. If None, all attributes will be used. Some
            languages only support "TOXICITY,". For more options, check Perspective's
            documentation. Defaults to None.
        lmbda (float, optional): Lambda/alpha parameter for Goodtriever/DExperts. Defaults to 2.0.
        knn_temp (int, optional): Temperature parameter for Goodtriever. Defaults to 100.
        filter_p (float, optional): top-p filter before ensembling. Defaults to 0.9.
        learning_rate (float, optional): Learning rate for DExperts. Defaults to 5e-5.

    Raises:
        NotImplementedError: If you set an invalid `kind`.

    """
    # Folder setup
    output_folder = setup_output_folder(
        output_folder,
        toxicity_choices,
        experiment_name,
        model_name,
        kind,
    )
    if multitask:
        if kind == "dexperts":
            multitask_files = {
                "toxic": tempfile.NamedTemporaryFile(suffix=".json", mode="w+"),
                "nontoxic": tempfile.NamedTemporaryFile(suffix=".json", mode="w+"),
            }

    # Logger setup
    (output_folder / "logs").mkdir(parents=True, exist_ok=True)
    # Check which domains were already processed if the command has been run before
    # This will be used to skip training if it was already done before.
    done_domains = (
        open(output_folder / "logs/done_domains.txt").readlines()
        if (output_folder / "logs/done_domains.txt").exists()
        else []
    )
    configure_logger(output_folder / "logs/experiment.log")

    # Domains setup
    train_folder = Path(train_folder)
    files = {
        "toxic": sorted(list(train_folder.glob(toxic_pattern))),
        "nontoxic": sorted(list(train_folder.glob(nontoxic_pattern))),
    }
    pretrained = {"toxic": pretrained_toxic, "nontoxic": pretrained_nontoxic}
    domains = domains or extract_domains_from_file_list(files["toxic"])
    domains = [str(d) for d in domains]
    logger.info(f"Domains: {', '.join(domains)}")

    logger.info(f"{'====' * 5}")
    logger.info(f"Starting '{output_folder}' experiment.")
    logger.info(f"{'====' * 5}")

    paths = defaultdict(lambda: None)
    try:
        for d, domain in enumerate(domains):
            for toxicity in toxicity_choices:
                # Check which domains were already added
                done = False
                if any(f"{domain}, {toxicity}" in done for done in done_domains):
                    logger.info(f"Skipped {domain}, {toxicity} build or train")
                    done = True

                logger.info(
                    f"Domain ({d}/{len(domains)}): {domain} // Toxicity: {toxicity}"
                )

                try:
                    file = [f for f in files[toxicity] if domain in str(f.name)][0]
                except IndexError:
                    # This will skip dstore building, but will return the pretrained path if any
                    logger.info(
                        f"No train files found for {domain}-{toxicity}. Using {pretrained[toxicity]}"
                    )
                    pretrained[toxicity] = pretrained[toxicity]
                    paths[toxicity] = pretrained[toxicity]
                    continue

                if multitask and kind in ["dexperts"]:
                    logger.info(f"Adding file: {file}")
                    curr_df = pd.read_json(file)

                    # temporary
                    if dstore_size is not None:
                        curr_df = curr_df.sample(dstore_size, random_state=42)

                    logger.info(f"Current number of samples: {curr_df.shape[0]}")

                    if d > 0:
                        previous_df = pd.read_json(multitask_files[toxicity].name)
                        logger.info(
                            f"Previously saved number of samples: {previous_df.shape[0]}"
                        )
                        curr_df = pd.concat([previous_df, curr_df], axis=0)
                        curr_df = curr_df.reset_index(drop=True)
                        logger.info(f"New number of samples: {curr_df.shape[0]}")

                    curr_df.to_json(multitask_files[toxicity].name, orient="records")
                    # Read file with appended data instead of unique domain ones
                    file = multitask_files[toxicity].name

                # We have to run even if done=True to get the dstore/model path
                model_path = (
                    output_folder / f"domain={d}-{domain}/checkpoints/{toxicity}"
                )
                if kind == "knn":
                    path = build_dstore(
                        output_folder=model_path,
                        train_file=file,
                        model_name=model_name,
                        toxicity=toxicity,
                        dstore_size=dstore_size,
                        pretrained_path=pretrained[toxicity],
                        log_folder=output_folder / f"logs/build_dstore_{toxicity}.log",
                        done=done,
                    )

                elif kind == "dexperts":
                    # always start from fresh pretrained model
                    if multitask:
                        pretrained[toxicity] = model_name
                    path = train_expert(
                        output_folder=model_path,
                        expert_name=f"finetune_{model_name}_{d}_{domain}_{toxicity}",
                        train_file=file,
                        model_name=model_name,
                        epochs=1,
                        learning_rate=learning_rate,
                        pretrained_path=pretrained[toxicity],
                        log_folder=output_folder / f"logs/train_{toxicity}.log",
                        done=done,
                    )
                else:
                    raise NotImplementedError(
                        "`kind` should be either 'knn' or 'dexperts' "
                    )

                # We'll have every intermediate model saved
                pretrained[toxicity] = path
                paths[toxicity] = path

                if not done:
                    with open(output_folder / "logs/done_domains.txt", "a") as f:
                        f.write(f"{domain}, {toxicity}\n")

            evaluate(
                domain=domain,
                output_folder=output_folder / f"domain={d}-{domain}",
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

    except KeyboardInterrupt:
        logger.info("Interrupted")


if __name__ == "__main__":
    fire.Fire(main)
