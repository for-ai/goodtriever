import json
from pathlib import Path
from typing import Any, Callable, Generator, List

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate(
    text: List[str],
    model: Callable,
    tokenizer: Callable,
    max_new_tokens: int,
    num_return_sequences: int,
    top_p: float,
) -> np.array:
    """Generate sequences given a prompt.

    Args:
        text (List[str]): Batch of prompts.
        model (Callable): HuggingFace model instance.
        tokenizer (Callable): HuggingFace tokenizer instance.
        max_new_tokens (int): Number of tokens to generate.
        num_return_sequences (int): Number of sequences to generate for each prompt.
        top_p (float): top_p probability for nucleus sampling.

    Returns:
        np.array: Prompt continuations
    """
    # Batched tokenization and generation
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=False).to(
        model.device
    )

    ## Nucleus sampling. As used in the RTP paper/repo.
    # "Top-p sampling chooses from the smallest possible set of words
    # whose cumulative probability exceeds the probability p."
    outputs = model.generate(
        **inputs,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=0,
    )
    continuations = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[-1] :],
        clean_up_tokenization_spaces=True,
        skip_special_tokens=True,
    )
    # Group generations from same prompt
    continuations = np.array(continuations).reshape((-1, num_return_sequences)).tolist()

    return continuations


def batched_generation(
    output_file: Path,
    prompts: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    num_return_sequences: int,
    max_new_tokens: int,
    top_p: float,
) -> Generator:
    """https://github.com/allenai/real-toxicity-prompts/blob/master/generation/generation.py#L61"""

    chunks = len(prompts) // batch_size
    print(f"Iterating on {chunks} chunks...")
    for chunk in tqdm(np.array_split(prompts, chunks), total=chunks):
        chunk = chunk["text"].values.tolist()

        continuations = generate(
            chunk,
            model,
            tokenizer,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )
        data = [{"prompt": p, "generations": c} for p, c in zip(chunk, continuations)]
        for d in data:
            with output_file.open("a") as f:
                print(json.dumps(d), file=f)
            yield d
