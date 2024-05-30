from pathlib import Path
from typing import Tuple

import fire
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

SUPPORTED_MODELS = ("nllb", "m2m")
# M2M lang codes = pt,fr,es,it,ru,ko,hi,ar
# NLLB lang codes = por_Latn,fra_Latn,spa_Latn,ita_Latn,rus_Cyrl,kor_Hang,hin_Deva,arb_Arab


def main(
    model_name: str,
    lang_code: Tuple[str],
    dataset: str = "data/jigsaw/toxicity_gte0.5_clean.json",
    output_dir="data/jigsaw/multilingual",
    batch_size=100,
    device_map="auto",
    load_in_4bit=False,
):
    """Translate a dataset to multiple languages using a pre-trained sequence-to-sequence model.

    Args:
        model_name (str): The name of the pre-trained model from huggingface to be used for translation.
        lang_code (Tuple[str]): A tuple of language codes specifying the target languages for translation.
        dataset (str, optional): The path to the dataset containing the text data to be translated.
            Defaults to "data/jigsaw/toxicity_gte0.5_clean.json".
        output_dir (str, optional): The directory where the translated data will be saved.
            Defaults to "data/jigsaw/multilingual".
        batch_size (int, optional): The number of samples to process in each batch.
            Defaults to 100.
        device_map (str, optional): The device mapping for model execution. Defaults to "auto".
        load_in_4bit (bool, optional): Flag indicating whether to load the model in 4-bit precision.
            Defaults to False.

    Raises:
        NotImplementedError: If the specified `model_name` is not supported.

    """
    if not any(m in model_name for m in SUPPORTED_MODELS):
        raise NotImplementedError(
            f"`model_name` not supported. Choose one of {', '.join(SUPPORTED_MODELS)}"
        )

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, device_map=device_map, quantization_config=bnb_config
    )

    df = pd.read_json(dataset)

    for code in lang_code:
        print(f"Translating to: {code}")

        forced_bos_token_id = (
            tokenizer.lang_code_to_id[code]
            if any(m in model_name for m in ("m2m", "nllb"))
            else None
        )

        all_translated = []
        for i in tqdm(range(0, len(df), batch_size), total=len(df) // batch_size):
            batch = df.iloc[i : i + batch_size]
            inputs = tokenizer(
                list(batch["text"].values),
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(model.device)

            translated_tokens = model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id, max_length=200
            )
            translated = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )

            all_translated.extend(translated)

        translated_df = pd.DataFrame({"text": all_translated})

        filename = dataset.split("/")[-1]
        output_file = Path(output_dir) / f"{code}_{filename}"
        print(f"Saving to: {output_file}")
        translated_df.to_json(output_file, orient="records")


if __name__ == "__main__":
    fire.Fire(main)
