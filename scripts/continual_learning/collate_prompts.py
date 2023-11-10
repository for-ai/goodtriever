from pathlib import Path

import fire
import pandas as pd

from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.perspective_api import unpack_scores


def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        if isinstance(response, dict):
            response = unpack_scores(response)[0]
        else:
            response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        yield {"text": generation, **response}


def main(
    original_prompts, scores, output_folder=None, helm=False, treat_as_prompts=True
):
    df_original = pd.read_json(original_prompts, lines=True)
    if treat_as_prompts:
        df_prompt = pd.json_normalize(df_original["prompt"])
    else:
        df_prompt = df_original

    df_scores = pd.read_json(scores, lines=True)

    # "Collate" prompt rescored results
    texts = df_prompt["text"].tolist()
    responses = df_scores["response"].tolist()
    generations_col_iter = make_generations_col(texts, responses)
    df_scores = pd.DataFrame(list(generations_col_iter))

    df_output = df_original.copy()
    del df_output["prompt"]
    df_output["prompt"] = df_scores.to_dict(orient="records")
    # df_output = pd.DataFrame({"prompt": scores.to_dict(orient="records")})

    if helm:
        # Add columns from HELM prompts
        if "id" in df_prompt.columns:
            df_output["id"] = df_prompt["id"]
        if "toxicity" in df_prompt.columns:
            df_output["original_toxicity"] = df_prompt["toxicity"]
        if "split" in df_prompt.columns:
            df_output["split"] = df_prompt["split"]

    if output_folder is None:
        output_folder = Path(scores).parent

    output_folder = output_folder / Path(original_prompts).name
    print(f"Writing to {output_folder}")
    df_output.to_json(output_folder, orient="records", lines=True)


if __name__ == "__main__":
    fire.Fire(main)
