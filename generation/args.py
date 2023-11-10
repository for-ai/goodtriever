from dataclasses import dataclass, field

from transformers import HfArgumentParser

from generation.knn_transformers.knnlm import DIST, KEY_TYPE


class GenerationParser:
    """Handle arguments involved in the generation process (kNN-LM + generate.py script)."""

    def __init__(self):
        parser = HfArgumentParser((GenerationArguments, KNNArguments))
        (
            self.gen_args,
            self.knn_args,
            self.other_strings,
        ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

        self.all_args = {
            "GenerationArguments": vars(self.gen_args),
            "KNNArguments": vars(self.knn_args),
            "OtherStrings": self.other_strings,
        }


@dataclass
class GenerationArguments:
    """
    Arguments pertaining general generation arguments.
    """

    output_folder: str
    prompts_path: str = field(
        default="gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl",
        metadata={"help": "Prompts filename."},
    )
    model_name: str = field(
        default="gpt2-large",
        metadata={
            "help": "Model to use from HuggingFace Hub. " "Defaults to 'gpt2-large'. "
        },
    )
    num_return_sequences: int = field(
        default=25,
        metadata={
            "help": "Number of sequences to return for each prompt. "
            "Defaults to 25. If `use_eos`, hard-coded to 1."
        },
    )
    max_new_tokens: int = field(
        default=20, metadata={"help": "Number of tokens to generate. Defaults to 20."}
    )
    top_p: float = field(
        default=0.90,
        metadata={
            "help": "Top-p for nucleus sampling (after ensemble). Defaults to 0.90."
        },
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Tokenization and generation batch size. Defaults to 16."},
    )
    output_filename: str = field(
        default=None,
        metadata={
            "help": "Output filename. If None, will be built "
            "automatically from user parameters. Defaults to None."
        },
    )
    num_prompts: int = field(
        default=None,
        metadata={"help": "Number of prompts to use. If None, will use all."},
    )


@dataclass
class KNNArguments:
    """
    Arguments pertaining retrieval-augmentation with kNN. Also supports DExperts experiments.
    """

    knn: bool = field(
        default=False, metadata={"help": "To retrieve or not to retrieve."}
    )
    dexperts: bool = field(
        default=False,
        metadata={
            "help": "To use DExperts or not. If `knn` is True, that supersedes this argument."
        },
    )
    dstore_size: int = field(
        default=None,
        metadata={
            "help": "The size of the dstore. If None (recommended), it will be detected automatically."
        },
    )
    knn_gpu: bool = field(
        default=True, metadata={"help": "To run kNN search on GPU or not."}
    )
    knn_keytype: KEY_TYPE.from_string = field(
        default=KEY_TYPE.last_ffn_input,
        metadata={
            "help": "The Key Type points to the layer to extract the datastore key from."
        },
    )
    dstore_dir: str = field(
        default="checkpoints",
        metadata={"help": "The directory of the first dstore (toxic)."},
    )
    other_dstore_dir: str = field(
        default=None,
        metadata={"help": "The directory of the second dstore (non-toxic)."},
    )
    k: int = field(
        default=1024,
        metadata={"help": "The number of retrieved neighbors from the first dstore."},
    )
    other_k: int = field(
        default=None,
        metadata={"help": "The number of retrieved neighbors from the second dstore."},
    )
    knn_sim_func: DIST.from_string = field(
        default=DIST.l2, metadata={"help": "kNN search similarity function."}
    )
    lmbda: float = field(
        default=2.0, metadata={"help": "Ensemble coefficient (alpha in the paper)."}
    )
    knn_temp: float = field(
        default=100,
        metadata={"help": "kNN temperature. The larger, the flatter the distribution."},
    )
    build_index: bool = field(
        default=False,
        metadata={"help": "If True, we'll train/save the FAISS index to disk."},
    )
    save_knnlm_dstore: bool = field(
        default=False, metadata={"help": "If True, saves datastores to disk."}
    )
    flat_index: bool = field(
        default=False,
        metadata={
            "help": "To use a flat index instead of the quantized (FAISS). "
            "Flat index is slower, but is totally accurate."
        },
    )
    ncentroids: int = field(
        default=4096,
        metadata={"help": "Number of centroids for quantized FAISS index."},
    )
    code_size: int = field(
        default=64,
        metadata={
            "help": "The code_size is typically a power of two between 4 and 64."
        },
    )
    probe: int = field(default=32, metadata={"help": "Number of probes at query time."})
    num_keys_to_add_at_a_time: int = field(
        default=1000000, metadata={"help": "Number of keys to add to index at a time."}
    )
    move_dstore_to_mem: bool = field(
        default=True, metadata={"help": "If True, loads keys and values to memory."}
    )
    no_load_keys: bool = field(
        default=True,
        metadata={
            "help": "If True, keys are not loaded to memory. Supersedes `move_dstore_to_mem`."
        },
    )
    recompute_dists: bool = field(
        default=False,
        metadata={
            "help": "Recompute the distances of retrieved neighbors for accurate probability computation."
        },
    )
    method: str = field(
        default="ensemble",
        metadata={
            "help": "Ensemble method. Choices: `ensemble`, `interpolate`, `interpolate_discourage`"
        },
    )
    filter_p: float = field(
        default=0.9, metadata={"help": "Top-p filtering (before ensemble)."}
    )
    ensemble_order: tuple = field(
        default=("subtract", "add"),
        metadata={
            "help": "Order of datastores on ensemble equation. By default, the first is subtracted."
        },
    )
    debug: bool = field(default=False, metadata={"help": "Turn on debug outputs."})
