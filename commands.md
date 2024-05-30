
# multilingual-toxicity-commands

## Raw mGPT - Baseline
```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_all --output_folder outputs/experiments/continual_learning/multilingual/holistic_toxicity/base_models/original --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged.jsonl --model_name "ai-forever/mGPT" --batch_size 4 --knn False --perspective_rate_limit 90 --custom_attrs TOXICITY, --perplexity_model ai-forever/mGPT --group_results_by lang
```

## Continual Learning experiments

### Goodtriever

#### High-resource

##### In-language, high resource
```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 90 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/goodtriever-multitask-1.3B/ --train_folder data/jigsaw/multilingual/domains/ --batch_size 4 --group_toxicity_by lang --toxicity_choices toxic,nontoxic --domains en,ru,it,fr,pt,es --toxic_pattern "jigsaw_*_toxic.json" --nontoxic_pattern "jigsaw_*_nontoxic.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind knn --knn_temp 200 --lmbda 2.0 --group_results_by lang
```

##### Translated, high-resource
```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 90 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/goodtriever-multitask-1.3B/translated --train_folder data/jigsaw/multilingual/minimal/ --batch_size 4 --group_toxicity_by lang --toxicity_choices toxic,nontoxic --domains en,rus,ita,fra,por,spa --toxic_pattern "*_gte0.5_clean_3000.json" --nontoxic_pattern "*_eq0_half_clean_10000.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind knn --knn_temp 200 --lmbda 2.0 --group_results_by lang
```


#### Translated, mid-resource

##### NLLB 600M - medium quality
```bash
# Mid-resource languages
CUDA_VISIBLE_DEVICES=0 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 180 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged_mid.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/goodtriever-multitask-1.3B/translated_mid --train_folder data/jigsaw/multilingual/minimal/ --batch_size 4 --toxicity_choices toxic,nontoxic --domains arb,kor,hin,por,rus,en --toxic_pattern "*_gte0.5_clean_3000.json" --nontoxic_pattern "*_eq0_half_clean_10000.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind knn --knn_temp 200 --lmbda 2.0 --group_results_by lang
```

##### M2M 418M - lower quality
```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 180 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged_mid.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/goodtriever-multitask-1.3B/translated_mid/m2m_418M --train_folder data/jigsaw/multilingual/minimal/m2m/418M --batch_size 4 --toxicity_choices toxic,nontoxic --domains ar,ko,hi,pt,ru,en --toxic_pattern "*_gte0.5_clean_3000.json" --nontoxic_pattern "*_eq0_half_clean_10000.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind knn --knn_temp 200 --lmbda 2.0 --group_results_by lang
```

##### NLLB 1.3B - higher quality
```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 20 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged_mid.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/goodtriever-multitask-1.3B/translated_mid/nllb_1.3b --train_folder data/jigsaw/multilingual/minimal/nllb1.3b --batch_size 4 --toxicity_choices toxic,nontoxic --domains arb,kor,hin,por,rus,en --toxic_pattern "*_gte0.5_clean_3000.json" --nontoxic_pattern "*_eq0_half_clean_10000.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind knn --knn_temp 200 --lmbda 2.0 --group_results_by lang
```

##### Different samples for each language (unparallel data experiments) - nllb 600m

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 20 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged_mid.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/goodtriever-multitask-1.3B/translated_mid/different_subsets --train_folder data/jigsaw/multilingual/different_subsets --batch_size 4 --toxicity_choices toxic,nontoxic --domains arb,kor,hin,por,rus,en --toxic_pattern "*_gte0.5_clean_3000_sampled.json" --nontoxic_pattern "*_eq0_half_clean_10000_sampled.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TbuOXICITY, --filter_p 0.9 --multitask True --kind knn --knn_temp 200 --lmbda 2.0 --group_results_by lang
```

##### 13B - medium quality translations

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 50 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged_mid.jsonl --model_name "ai-forever/mGPT-13B" --experiment_name multilingual/holistic_toxicity/original/jigsaw/goodtriever-multitask-13B/translated_mid --train_folder data/jigsaw/multilingual/minimal/ --batch_size 2 --toxicity_choices toxic,nontoxic --domains arb,kor,hin,por,rus,en --toxic_pattern "*_gte0.5_clean_3000.json" --nontoxic_pattern "*_eq0_half_clean_10000.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind knn --knn_temp 200 --lmbda 2.0 --group_results_by lang
```

### DExperts

#### High-resource

##### In-language, high-resource
```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 180 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/dexperts-multitask-1.3B/ --train_folder data/jigsaw/multilingual/domains/ --batch_size 4 --toxicity_choices toxic,nontoxic --domains en,ru,it,fr,pt,es --toxic_pattern "jigsaw_*_toxic.json" --nontoxic_pattern "jigsaw_*_nontoxic.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind dexperts --lmbda 2.0 --group_results_by lang --learning_rate 5e-6
```

##### In-language, high-resource, inverted order
```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 180 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/dexperts-multitask-1.3B/native_inverted --train_folder data/jigsaw/multilingual/domains/ --batch_size 4 --toxicity_choices toxic,nontoxic --domains es,pt,fr,it,ru,en --toxic_pattern "jigsaw_*_toxic.json" --nontoxic_pattern "jigsaw_*_nontoxic.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind dexperts --lmbda 2.0 --group_results_by lang --learning_rate 5e-6
```

##### Translated, high-resource
```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 180 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/dexperts-multitask-1.3B/translated --train_folder data/jigsaw/multilingual/minimal/ --batch_size 4 --toxicity_choices toxic,nontoxic --domains en,rus,ita,fra,por,spa --toxic_pattern "*_gte0.5_clean_3000.json" --nontoxic_pattern "*_eq0_half_clean_10000.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind dexperts --lmbda 2.0 --group_results_by lang --learning_rate 5e-6
```


#### Translated, mid-resource

##### NLLB 600M - Medium quality
```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 50 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged_mid.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/dexperts-multitask-1.3B/translated_mid/redo --train_folder data/jigsaw/multilingual/minimal/ --batch_size 4 --toxicity_choices toxic,nontoxic --domains arb,kor,hin,por,rus,en --toxic_pattern "*_gte0.5_clean_3000.json" --nontoxic_pattern "*_eq0_half_clean_10000.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind dexperts --lmbda 2.0 --group_results_by lang --learning_rate 5e-6
```

##### M2M 418M - Lower quality
```bash
CUDA_VISIBLE_DEVICES=1 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 180 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged_mid.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/dexperts-multitask-1.3B/translated_mid/m2m_418M_1e-6 --train_folder data/jigsaw/multilingual/minimal/m2m/418M --batch_size 4 --toxicity_choices toxic,nontoxic --domains ar,ko,hi,pt,ru,en --toxic_pattern "*_gte0.5_clean_3000.json" --nontoxic_pattern "*_eq0_half_clean_10000.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind dexperts --lmbda 2.0 --group_results_by lang --learning_rate 1e-6
```

##### Different samples for each language (unparallel data experiments) - nllb 600m
```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.continual_learning.cl_experiment --perspective_rate_limit 50 --prompts_path data/holistic_toxicity/original/v1.1/holistic_toxicity_merged_mid.jsonl --model_name "ai-forever/mGPT" --experiment_name multilingual/holistic_toxicity/original/jigsaw/dexperts-multitask-1.3B/translated_mid/different_subsets --train_folder data/jigsaw/multilingual/different_subsets --batch_size 4 --toxicity_choices toxic,nontoxic --domains arb,kor,hin,por,rus,en --toxic_pattern "*_gte0.5_clean_3000_sampled.json" --nontoxic_pattern "*_eq0_half_clean_10000_sampled.json" --perplexity_model "ai-forever/mGPT" --custom_attrs TOXICITY, --filter_p 0.9 --multitask True --kind dexperts --lmbda 2.0 --group_results_by lang --learning_rate 5e-6
```


## Run translation script
```bash
#NLLB 1.3b
CUDA_VISIBLE_DEVICES=0 python -m scripts.utils.translate --model_name facebook/nllb-200-distilled-1.3B --lang_code arb_Arab,hin_Deva,kor_Hang,rus_Cyrl,por_Latn --output_dir data/jigsaw/multilingual/minimal/nllb1.3b --dataset data/jigsaw/multilingual/minimal/en_toxicity_eq0_half_clean_10000.json --batch_size 8
```


## Datastore size experiments

### Goodtriever
```bash
# vary toxic tokens
CUDA_VISIBLE_DEVICES=0 python -m experiments.datastore_size_experiment --toxic_tokens 100000,250000,500000,1000000,5000000,None --nontoxic_tokens None --experiment_name datastore_size/pt/goodtriever --perspective_rate_limit 80 --model_name ai-forever/mgpt --perplexity_model ai-forever/mgpt --toxic_train_file data/jigsaw/multilingual/por_Latn_toxicity_gte0.5_clean.json --nontoxic_train_file data/jigsaw/multilingual/por_Latn_toxicity_eq0_half_clean.json --knn_temp 200 --prompts_path data/holistic_toxicity/original/v1.1/sentences_sampled_pt-BR.jsonl --custom_attrs TOXICITY, --batch_size 2

# vary nontoxic tokens
CUDA_VISIBLE_DEVICES=1 python -m experiments.datastore_size_experiment --nontoxic_tokens 100000,250000,500000,1000000,5000000,10000000,20000000,None --toxic_tokens None --experiment_name datastore_size/pt/goodtriever --perspective_rate_limit 80 --model_name ai-forever/mgpt --perplexity_model ai-forever/mgpt --toxic_train_file data/jigsaw/multilingual/por_Latn_toxicity_gte0.5_clean.json --nontoxic_train_file data/jigsaw/multilingual/por_Latn_toxicity_eq0_half_clean.json --knn_temp 200 --prompts_path data/holistic_toxicity/original/v1.1/sentences_sampled_pt-BR.jsonl --custom_attrs TOXICITY, --batch_size 2
```
