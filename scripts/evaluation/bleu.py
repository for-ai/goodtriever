from typing import List, Union

import fire
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

BLEU_TOKENIZERS = {"chinese": "zh", "japanese": "ja-mecab", "korean": "ko-mecab"}


def _sacre_bleu_score(
    labels: List[str], generations: List[str], scorer: Union[BLEU, CHRF]
):
    """Compute BLEU score using sacreBLEU library.

    Args:
        labels: list of reference texts
        generations: list of generated texts
        scorer: BLEU or CHRF scorer from sacrebleu library

    """
    hyps, refs = [], [[]]
    for l, g in zip(labels, generations):
        l, g = l.strip(), g.strip()
        refs[0].append(l)
        hyps.append(g)
    result = scorer.corpus_score(hyps, refs)
    return result


def bleu_corpus_score_special_tokenizer(
    labels: List[str], generations: List[str], target_language: str = "English"
):
    """BLEU score with a special tokenizer for Japanese, Korean, and Chinese.

    Args:
        labels: list of reference texts
        generations: list of generated texts
        target_language: language name. Makes sense to specify only Korean, Japanese or Chinese, otherwise will use the default tokenizer.
    """
    tokenizer = BLEU_TOKENIZERS.get(target_language.lower(), "13a")
    bleu = BLEU(tokenize=tokenizer)
    result = _sacre_bleu_score(labels, generations, bleu)
    return {"BLEU_corpus_score": round(result.score, 3)}


def chrf_pp_corpus_score(labels: List[str], generations: List[str]):
    """ChrF++ to be more precise"""
    chrf = CHRF(word_order=2)
    result = _sacre_bleu_score(labels, generations, chrf)
    return {"ChrF_corpus_score": round(result.score, 3)}


def main(language: str, labels_file: str, generations_file: str):
    """Computes BLEU score for a given language"""
    labels = pd.read_json(labels_file)["text"].tolist()
    generations = pd.read_json(generations_file)["text"].tolist()

    bleu = bleu_corpus_score_special_tokenizer(labels, generations, language)
    chrf = chrf_pp_corpus_score(labels, generations)

    print(f"BLEU score for {language} is {bleu}")
    print(f"ChrF++ score for {language} is {chrf}")


if __name__ == "__main__":
    fire.Fire(main)
