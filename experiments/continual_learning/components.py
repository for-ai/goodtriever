import subprocess
import logging

from experiments.continual_learning.utils import run

logger = logging.getLogger(__name__)


def build_dstore(
    output_folder,
    train_file,
    model_name,
    toxicity,
    dstore_size=None,
    pretrained_path=None,
    flat_index=False,
    done=False,
    log_folder="logs",
):
    output_folder.mkdir(exist_ok=True, parents=True)

    if done:
        return output_folder

    if pretrained_path is not None:
        logger.info(
            f"Copying base {toxicity} dstore from {pretrained_path} to {output_folder}."
        )
        subprocess.run(f"cp -r {pretrained_path}/* {output_folder}/", shell=True)
        logger.info(f"Base {toxicity} dstore copied.")

    ds_cmd = f"""
        python -u -m generation.knn_transformers.run_clm \
            --model_name_or_path {model_name} \
            --train_file {train_file} \
            --eval_subset train \
            --output_dir {output_folder} \
            --dstore_dir {output_folder} \
            --save_knnlm_dstore \
            --continue_writing \
            {f'--dstore_size {dstore_size} --limit_eval_to_dstore' if dstore_size else ''} \
            --do_eval | tee -a {log_folder}
    """

    train_cmd = f"""
        python -u -m generation.knn_transformers.run_clm \
            --model_name_or_path {model_name} \
            --train_file {train_file} \
            --eval_subset train \
            --output_dir {output_folder} \
            --dstore_dir {output_folder} \
            {f'--dstore_size {dstore_size}' if dstore_size else ''} \
            {'--flat_index' if flat_index else ''} \
            --build_index | tee -a {log_folder}
    """

    logger.info(f"Running `datastore build` command: {ds_cmd}")
    run(ds_cmd)

    logger.info(f"Running `index train` command: {train_cmd}")
    run(train_cmd)

    return output_folder


def train_expert(
    output_folder,
    expert_name,
    train_file,
    model_name,
    pretrained_path=None,
    epochs=1,
    block_size=128,
    batch_size=4,
    grad_accum=16,
    log_folder="logs/",
    done=False,
):
    """Train DExperts expert (either toxic or non-toxic)."""
    expert_path = output_folder / model_name / expert_name

    if done:
        return expert_path

    if pretrained_path is not None:
        model_name = pretrained_path

    train_cmd = f"""
        python -m scripts.finetuning.finetune_gpt2 \
            --output_dir {expert_path} \
            --model_type gpt2 \
            --model_name_or_path {model_name} \
            --do_train \
            --num_train_epochs {epochs} \
            --block_size {block_size} \
            --save_total_limit 1 \
            --dataloader_drop_last \
            --per_device_train_batch_size {batch_size} \
            --gradient_accumulation_steps {grad_accum} \
            --train_data_file {train_file} \
            --overwrite_cache | tee -a {log_folder}
    """
    run(train_cmd)

    return expert_path
