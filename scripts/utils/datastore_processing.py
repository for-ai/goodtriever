from pathlib import Path

import fire
import numpy as np
import pandas as pd
from datasets import load_dataset


def main(
    output_folder: str,
    use_custom: bool = False,
    use_paradetox: bool = False,
    use_toxigen: bool = False,
    rtp_path: str = "data/rescored/rtp_full_sequences_filtered_dexperts_collated.jsonl",
    toxic_threshold: float = 0.5,
    nontoxic_threshold: float = 0.1,
):
    """Process common datasets to be used as datastores.

    Data is saved as toxic and non-toxic json files.

    Currently, this script supports processing for:
        - RealToxicityPrompts (as custom)
        - Paradetox
        - Toxigen

    If multiple datasets are processed in a same run, the final json
    files contains data from them all.

    Args:
        output_folder (str): Folder to save processed json files to.
        use_custom (bool, optional): Whether to process custom dataset. Defaults to False.
        use_paradetox (bool, optional): Whether to process paradetox. Defaults to False.
        use_toxigen (bool, optional): Whether to process Toxigen. Defaults to False.
        custom_jsonl_path (str, optional): Path to custom jsonl file.
            Defaults to "data/rescored/rtp_full_sequences_filtered_dexperts_collated.jsonl".
        toxic_threshold (float, optional): Samples higher than this threshold are toxic.
            Defaults to 0.5.
        nontoxic_threshold (float, optional): Samples lower or equal to this threshold are non-toxic.
            Defaults to 0.1.
    """
    toxic_ds = pd.DataFrame()
    nontoxic_ds = pd.DataFrame()

    if use_custom:
        custom = pd.read_json(rtp_path, lines=True)
        toxic = custom[custom["toxicity"] > toxic_threshold]
        nontoxic = custom[custom["toxicity"] <= nontoxic_threshold]

        toxic_ds = pd.concat([toxic_ds, pd.DataFrame(toxic["text"], columns=["text"])])
        nontoxic_ds = pd.concat(
            [nontoxic_ds, pd.DataFrame(nontoxic["text"], columns=["text"])]
        )

    if use_paradetox:
        dataset = load_dataset("s-nlp/paradetox")
        toxic = dataset["train"]["en_toxic_comment"]
        nontoxic = dataset["train"]["en_neutral_comment"]

        toxic_ds = pd.concat([toxic_ds, pd.DataFrame(toxic, columns=["text"])])
        nontoxic_ds = pd.concat([nontoxic_ds, pd.DataFrame(nontoxic, columns=["text"])])

    if use_toxigen:
        dataset = load_dataset("skg/toxigen-data", name="train", use_auth_token=True)[
            "train"
        ]  # 250k training examples

        toxic_mask = np.array(dataset["prompt_label"]) >= toxic_threshold
        nontoxic_mask = np.array(dataset["prompt_label"]) < nontoxic_threshold

        prompts = np.array(dataset["prompt"])
        toxic = np.stack(
            [s for p in prompts[toxic_mask] for s in p.split("\\n- ")]
        ).reshape(-1)
        nontoxic = np.stack(
            [s for p in prompts[nontoxic_mask] for s in p.split("\\n- ")]
        ).reshape(-1)
        toxic = pd.DataFrame(toxic, columns=["text"]).drop_duplicates()
        nontoxic = pd.DataFrame(nontoxic, columns=["text"]).drop_duplicates()

        gens = np.array(dataset["generation"])

        toxic = pd.concat([toxic, pd.DataFrame(gens[toxic_mask], columns=["text"])])
        nontoxic = pd.concat(
            [nontoxic, pd.DataFrame(gens[nontoxic_mask], columns=["text"])]
        )

        toxic["text"] = (
            toxic["text"].str.replace(r"\\n", "", regex=True).str.strip("-").str.strip()
        )
        nontoxic["text"] = (
            nontoxic["text"]
            .str.replace(r"\\n", "", regex=True)
            .str.strip("-")
            .str.strip()
        )

        toxic_ds = pd.concat([toxic_ds, toxic])
        nontoxic_ds = pd.concat([nontoxic_ds, nontoxic])

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    for mode, dataset in zip(["toxic", "nontoxic"], [toxic_ds, nontoxic_ds]):
        name = mode
        if use_paradetox:
            name += "_paradetox"
        if use_custom:
            name += "_custom"
        output_file = output_folder / f"{name}.json"

        dataset.to_json(output_file, orient="records")

    print(f"json files saved at {output_folder}")


if __name__ == "__main__":
    fire.Fire(main)
