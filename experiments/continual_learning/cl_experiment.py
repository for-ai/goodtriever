import logging
import tempfile
import fire
import pandas as pd

from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

from experiments.continual_learning.components import (
    build_dstore,
    train_expert,
)
from experiments.continual_learning.utils import (
    evaluate,
    extract_domains_from_file_list,
    setup_output_folder,
)
from experiments.logger import configure_logger

logger = logging.getLogger(__name__)


def main(
    domains: Optional[Tuple] = None,
    model_name: str = "gpt2-large",
    toxicity_choices: Tuple = ("toxic", "nontoxic"),
    prompts_path: str = "data/continual_mitigation/prompts/wilds_5_clusters_200_samples_toxic.jsonl",
    train_folder: str = "data/continual_mitigation/domains/train",
    output_folder: str = "outputs/experiments/continual_learning",
    experiment_name: str = "continual_mitigation",
    perspective_rate_limit: str = 90,
    batch_size: int = 4,
    group_toxicity_by: str = "domain",
    kind: str = "knn",
    pretrained_toxic: Optional[str] = None,
    pretrained_nontoxic: Optional[str] = None,
    dstore_size: Optional[int] = None,
    num_prompts: Optional[int] = None,
    multitask: Optional[bool] = False,
):
    """Run continual learning experiments.

    Args:
        domains (Optional[Tuple], optional): The list of domains. If None,
            they will be inferred from files. Defaults to None.
        model_name (str, optional): Base model. Defaults to "gpt2-large".
        toxicity_choices (Tuple, optional): Which datastores will be continually improved.
            Defaults to ("toxic", "nontoxic").
        prompts_path (str, optional): Path to main prompts file.
            Defaults to "data/continual_mitigation/prompts/wilds_5_clusters_200_samples_toxic.jsonl".
        train_folder (str, optional): Folder that contains all training files.
            Defaults to "data/continual_mitigation/domains/train".
        output_folder (str, optional): Folder to save experiments output.
            Defaults to "outputs/experiments/continual_learning".
        experiment_name (str, optional): Experiment name will create a new folder inside `output_folder`.
            Defaults to "continual_mitigation".
        perspective_rate_limit (str, optional): Perspective API rate limit. Defaults to 90.
        batch_size (int, optional): Batch size for inference. Defaults to 4.
        group_toxicity_by (str, optional): This column should be in the prompts file and
            it is used to compute domain-specific toxicity metrics. Defaults to "domain".
        kind (str, optional): Options are ('knn', 'dexperts'). Defaults to "knn".
        pretrained_toxic (Optional[str], optional): Path to a pretrained
            toxic datastore or expert. Defaults to None.
        pretrained_nontoxic (Optional[str], optional): Path to a pretrained
            nontoxic datastore or expert. Defaults to None.
        dstore_size (Optional[int], optional): Set if you want to limit datastore size.
            Defaults to None.
        num_prompts (Optional[int], optional): Number of prompts to be evaluated.
            If None, will use all. Defaults to None.
        multitask (Optional[bool], optional): If True, DExperts will be run
            in the continual learning manner. At every step, experts will be
            trained with all data available. Defaults to False.

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
        "toxic": sorted(list(train_folder.glob("wilds_*_toxic.json"))),
        "nontoxic": sorted(list(train_folder.glob("wilds_*_nontoxic.json"))),
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
                    file = [f for f in files[toxicity] if domain in str(f)][0]
                except IndexError:
                    # This will skip dstore building, but will return the pretrained path if any
                    logger.info(
                        f"No train files found for {domain}-{toxicity}. Using {pretrained[toxicity]}"
                    )
                    pretrained[toxicity] = pretrained[toxicity]
                    paths[toxicity] = pretrained[toxicity]
                    continue

                if multitask:
                    curr_df = pd.read_json(file)
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
                    path = train_expert(
                        output_folder=model_path,
                        expert_name=f"finetune_{model_name}_{d}_{domain}_{toxicity}",
                        train_file=file,
                        model_name=model_name,
                        epochs=1,
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
                perspective_rate_limit=perspective_rate_limit,
                group_toxicity_by=group_toxicity_by,
                num_prompts=num_prompts,
                kind=kind,
                batch_size=batch_size,
            )

    except KeyboardInterrupt:
        logger.info("Interrupted")


if __name__ == "__main__":
    fire.Fire(main)
