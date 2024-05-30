DATA_DIR=data/jigsaw/multilingual/minimal/dexperts_multi_debug
BATCH_SIZE=4
BLOCK_SIZE=128
GRAD_ACCUM_STEPS=16

WANDB_PROJECT=dexperts-training python -m scripts.finetuning.finetune_gpt2 \
	--output_dir models/experts/toxicity/large/finetuned_mgpt_en_rus_toxic_wandb \
	--model_type gpt2 \
	--model_name_or_path ai-forever/mGPT \
	--do_train \
	--num_train_epochs 1 \
	--block_size $BLOCK_SIZE \
	--save_total_limit 1 \
	--dataloader_drop_last \
	--per_device_train_batch_size $BATCH_SIZE \
	--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
	--train_data_file $DATA_DIR/en_rus_toxic_clean.json \
	--overwrite_cache \
	--report_to wandb \
	--logging_steps 10 \
	--run_name dexperts_en_rus_toxic
