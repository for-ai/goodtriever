import logging
import os
import re
import time
from enum import Enum, auto
from pathlib import Path

import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.setLevel(20)


class DIST(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()


class Datastore:
    def __init__(
        self,
        dstore_dir,
        dimension,
        model_type,
        device,
        knn_gpu=False,
        knn_sim_func=DIST.l2,
        move_dstore_to_mem=False,
        probe=32,
        no_load_keys=False,
        dstore_size=None,
        flat_index=False,
        continue_writing=False,
    ):
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.model_type = model_type
        self.device = device

        self.knn_gpu = knn_gpu
        self.knn_sim_func = knn_sim_func
        self.move_dstore_to_mem = move_dstore_to_mem
        self.probe = probe
        self.no_load_keys = no_load_keys
        self.flat_index = flat_index
        self.continue_writing = continue_writing

        self.dstore_size = dstore_size or self.get_dstore_size()
        self.previous_dstore_size = (
            self.get_dstore_size() if self.continue_writing else None
        )

        self.index = None
        self.keys = None
        self.vals = None

        dist_type_to_dist_func = {
            DIST.l2: Datastore.l2,
            DIST.dot: Datastore.dotprod,
        }
        self.dist_func = dist_type_to_dist_func[
            knn_sim_func
        ]  # l2 or dot product function

    def get_knns(self, queries, k=1024, recompute_dists=False):
        if not self.knn_gpu:
            queries = queries.cpu()
        dists, knns = self.index.search(queries, k)
        dists, knns = dists.to(self.device), knns.to(self.device)

        if recompute_dists:
            knns_vecs = torch.from_numpy(self.keys[knns.cpu().numpy()]).to(self.device)
            dists = self.dist_func(queries, knns_vecs)
        return dists, knns

    def knns_to_log_prob(
        self,
        knns,
        neg_dists,
        vocab_size,
        knn_temperature=1.0,
    ):
        probs = torch.nn.functional.softmax(neg_dists / knn_temperature, dim=-1)
        vals_at_knns = self.vals[knns].squeeze(-1)  # (nonpad batch * time, k)
        knn_log_probs = (
            torch.full(size=(vals_at_knns.shape[:-1] + (vocab_size,)), fill_value=0.0)
            .to(self.device)
            .scatter_add(dim=-1, index=vals_at_knns, src=probs)
            .log()
        )  # (nonpad_batch * time, vocab)
        knn_log_probs = torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0)
        return knn_log_probs, vals_at_knns

    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys) ** 2, dim=-1)

    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)

    def setup_faiss(self):
        if not self.dstore_dir:
            raise ValueError("Cannot build a datastore without the data.")

        start = time.time()
        index_name = self.get_index_path()

        if not Path(index_name).exists():
            if Path(index_name.replace("_flat", "")).exists():
                logger.info(
                    f"Fallback index {index_name} to its non-flat version, which exists."
                )
                self.flat_index = False
                index_name = index_name.replace("_flat", "")

        cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logger.info(f"Reading datastore took {time.time() - start} s")
        cpu_index.nprobe = self.probe

        if self.knn_gpu:
            start = time.time()
            co = faiss.GpuClonerOptions()
            if not self.flat_index:
                # This causes memory errors on large flat indexes
                co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, cpu_index, co
            )
            logger.info(f"Moving index to GPU took {time.time() - start} s")
        else:
            gpu_index = cpu_index

        # make_direct_map() allows calling reconstruct(n),
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        if not self.flat_index:
            cpu_index.make_direct_map()

        keys_vals_prefix = self.get_dstore_path()
        if not self.no_load_keys:
            self.keys = np.memmap(
                f"{keys_vals_prefix}_keys.npy",
                dtype=np.float16,
                mode="r",
                shape=(self.dstore_size, self.dimension),
            )
        self.vals = np.memmap(
            f"{keys_vals_prefix}_vals.npy",
            dtype=np.int32,
            mode="r",
            shape=(self.dstore_size, 1),
        )
        # self.vals = torch.from_numpy(self.vals).to(self.device)

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if self.move_dstore_to_mem:
            logger.info("Loading to memory...")
            start = time.time()

            if not self.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(
                    f"{keys_vals_prefix}_keys.npy",
                    dtype=np.float16,
                    mode="r",
                    shape=(self.dstore_size, self.dimension),
                )
                self.keys = self.keys_from_memmap[:].astype(np.float16)

            del self.vals
            vals_from_memmap = np.memmap(
                f"{keys_vals_prefix}_vals.npy",
                dtype=np.int32,
                mode="r",
                shape=(self.dstore_size, 1),
            )
            self.vals = torch.from_numpy(vals_from_memmap[:]).long().to(self.device)
            del vals_from_memmap
            logger.info("Loading to memory took {} s".format(time.time() - start))
        self.index = gpu_index
        return self

    def build_index(
        self,
        num_keys_to_add_at_a_time=1000000,
        ncentroids=4096,
        seed=1,
        code_size=64,
        probe=32,
    ):
        if self.previous_dstore_size is not None:
            if self.dstore_size != self.previous_dstore_size:
                raise RuntimeError(
                    "If training the index, disable `continue_writing` "
                    "or set `dstore_size` to None. Delete the datastores and start over."
                )
        logger.info("Building index")
        index_name = self.get_index_path()

        # Flat index may be used to debug retrieved results.
        # It returns a 100% accurate nearest neighbor search, while quantized
        # index are (faster, but often inaccurate) approximations of those.
        # https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing#the-indexidmap
        if self.flat_index:
            index = faiss.IndexFlatL2(self.dimension)
        else:
            # Initialize faiss index
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFPQ(
                quantizer, self.dimension, ncentroids, code_size, 8
            )
            index.nprobe = probe

            logger.info("Training Index")
            np.random.seed(seed)
            random_sample = np.random.choice(
                np.arange(self.dstore_vals.shape[0]),
                size=[min(1000000, self.dstore_vals.shape[0])],
                replace=False,
            )
            logger.info(f"Training samples: {random_sample.shape[0]}")
            start = time.time()
            # Faiss does not handle adding keys in fp16 as of writing this.
            index.train(self.dstore_keys[random_sample].astype(np.float32))
            logger.info(f"Training took {time.time() - start} s")

        logger.info("Adding Keys")
        # index = faiss.read_index(f'{index_name}.trained')
        # TODO Could this be enhanced for continual learning to be faster?
        start = 0
        start_time = time.time()
        while start < self.dstore_size:
            end = min(self.dstore_size, start + num_keys_to_add_at_a_time)
            to_add = self.dstore_keys[start:end].copy()
            if self.flat_index:
                index.add(torch.tensor(to_add.astype(np.float32)))
            else:
                index.add_with_ids(
                    torch.tensor(to_add.astype(np.float32)), torch.arange(start, end)
                )
            start += num_keys_to_add_at_a_time

            if (start % 1000000) == 0:
                logger.info(f"Added {start} tokens so far")
                logger.info(f"Writing Index {start}")
                faiss.write_index(index, f"{index_name}")

        logger.info(f"Adding total {start} keys")
        logger.info(f"Adding took {time.time() - start_time} s")
        logger.info(f"Writing Index to {index_name}")
        start = time.time()
        faiss.write_index(index, f"{index_name}")
        logger.info(f"Writing index took {time.time() - start} s")

    def load_keys_and_vals(self):
        keys_vals_prefix = self.get_dstore_path()
        keys_filename = f"{keys_vals_prefix}_keys.npy"
        vals_filename = f"{keys_vals_prefix}_vals.npy"
        if os.path.exists(keys_filename) and os.path.exists(vals_filename):
            mode = "r"
        else:
            mode = "w+"
            Path(keys_filename).parent.mkdir(parents=True, exist_ok=True)

        dstore_keys = np.memmap(
            keys_filename,
            dtype=np.float16,
            mode=mode,
            shape=(self.dstore_size, self.dimension),
        )
        dstore_vals = np.memmap(
            vals_filename, dtype=np.int32, mode=mode, shape=(self.dstore_size, 1)
        )

        # If continual addition of tokens is not enabled or we found no
        # previous dstore, the default behavior ir to just create/load the
        # dstore with current `self.dstore_size`.
        if self.continue_writing is False or self.previous_dstore_size is None:
            self.dstore_keys = dstore_keys
            self.dstore_vals = dstore_vals
            self._keys_path = keys_filename

        # If continual learning is enabled, we should load previous datastores
        # of `previous_dstore_size` and concatenate keys and values with
        # the new memmaps of `dstore_size`.
        elif self.continue_writing is True and self.previous_dstore_size is not None:
            # Load previous dstore
            previous_prefix = self.build_prefix(
                "dstore",
                self.dstore_dir,
                self.model_type,
                self.previous_dstore_size,
                self.dimension,
            )
            previous_keys_filename = f"{previous_prefix}_keys.npy"
            previous_vals_filename = f"{previous_prefix}_vals.npy"

            previous_keys = np.memmap(
                previous_keys_filename,
                dtype=np.float16,
                mode="r",
                shape=(self.previous_dstore_size, self.dimension),
            )
            previous_vals = np.memmap(
                previous_vals_filename,
                dtype=np.int32,
                mode="r",
                shape=(self.previous_dstore_size, 1),
            )

            # Load memmapped dstore with actual size after merging
            full_size = self.dstore_size + self.previous_dstore_size
            merged_prefix = self.build_prefix(
                "dstore", self.dstore_dir, self.model_type, full_size, self.dimension
            )
            merged_keys_filename = f"{merged_prefix}_keys.npy"
            merged_vals_filename = f"{merged_prefix}_vals.npy"

            self.dstore_keys = np.memmap(
                merged_keys_filename,
                dtype=np.float16,
                mode="w+",
                shape=(full_size, self.dimension),
            )
            self.dstore_vals = np.memmap(
                merged_vals_filename, dtype=np.int32, mode="w+", shape=(full_size, 1)
            )

            # Fill values
            self.dstore_keys[0 : self.previous_dstore_size] = previous_keys
            self.dstore_keys[self.previous_dstore_size :] = dstore_keys

            self.dstore_vals[0 : self.previous_dstore_size] = previous_vals
            self.dstore_vals[self.previous_dstore_size :] = dstore_vals

            del dstore_keys, dstore_vals, previous_keys, previous_vals

            # Delete previous files
            os.remove(keys_filename)
            os.remove(vals_filename)
            try:
                os.remove(previous_keys_filename)
                os.remove(previous_vals_filename)
            except FileNotFoundError:
                logger.debug("Are you trying to concatenate a file to itself?")

            # Clean indexes from the previous dstore size
            previous_index = Path(self.dstore_dir).glob(
                f"index_{self.model_type}_{self.previous_dstore_size}*"
            )
            for path in previous_index:
                os.remove(path)

            logger.info(
                f"Updated dstore_size from {self.previous_dstore_size} to {full_size} (+{self.dstore_size})."
            )
            self.dstore_size = full_size
            self._keys_path = merged_keys_filename

        return self

    def get_dstore_size(self, index=-1) -> int:
        """Return a datastore size from datastores found.

        Datastores names are sorted alphabetically, so they're expected
        to follow smaller -> largest number of tokens. They're also in
        pairs for each number of tokens (dstore keys and vals files).

        Args:
            index (int, optional): Index of datastore. Defaults to -1 (largest).

        Raises:
            ValueError: If `dstore_dir` does not exist.
            RuntimeError: If the number of datastores in `dstore_dir` ir not even.

        Returns:
            int: The size of the datastore in the index.
        """
        if not Path(self.dstore_dir).exists():
            raise ValueError(f"dstore_dir {self.dstore_dir} does not exist.")

        dstores = sorted(list(Path(self.dstore_dir).glob("dstore_*")))

        if not dstores:
            return None

        if len(dstores) % 2 != 0:
            raise RuntimeError("There should be a pair value of dstores in folder.")

        pattern = r"(?<=_)\d+(?=_)"
        # idx 0 = dstore size, idx 1 = dimension
        size = int(re.findall(pattern, dstores[index].stem)[0])
        return size

    @staticmethod
    def build_prefix(base, dstore_dir, model_type, dstore_size, dimension, flat=False):
        prefix = f"{dstore_dir}/{base}_{model_type}_{dstore_size}_{dimension}"

        if base == "index" and flat:
            prefix = f"{prefix}{'_flat' if flat else ''}"

        return prefix

    def get_dstore_path(self):
        return self.build_prefix(
            "dstore", self.dstore_dir, self.model_type, self.dstore_size, self.dimension
        )

    def get_index_path(self):
        prefix = self.build_prefix(
            "index",
            self.dstore_dir,
            self.model_type,
            self.dstore_size,
            self.dimension,
            flat=self.flat_index,
        )
        return f"{prefix}.indexed"
