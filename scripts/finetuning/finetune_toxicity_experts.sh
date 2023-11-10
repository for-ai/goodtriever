DATA_DIR=data/continual_mitigation/jigsaw
BATCH_SIZE=4
BLOCK_SIZE=128
GRAD_ACCUM_STEPS=16

python -m scripts.finetuning.finetune_gpt2 \
	--output_dir models/experts/toxicity/large/finetuned_gpt2_toxic_ours \
	--model_type gpt2 \
	--model_name_or_path gpt2-large \
	--do_train \
	--num_train_epochs 1 \
	--block_size $BLOCK_SIZE \
	--save_total_limit 1 \
	--dataloader_drop_last \
	--per_device_train_batch_size $BATCH_SIZE \
	--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
	--train_data_file $DATA_DIR/toxicity_gte0.5_no_dem.json \
	--overwrite_cache
