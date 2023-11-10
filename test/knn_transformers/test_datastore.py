import tempfile
from pathlib import Path

import numpy as np
import pytest

from generation.knn_transformers.datastore import Datastore


def test_datastore_continual_add():
    """Test if continual addition of dstore keys/vals works as expected."""
    with tempfile.TemporaryDirectory() as dstore_dir:
        # Test continual add
        size = 1000
        datastore = Datastore(
            dstore_dir=dstore_dir,
            dimension=10,
            model_type="model",
            device="cuda",
            flat_index=False,
            continue_writing=True,
            dstore_size=size,
        ).load_keys_and_vals()
        datastore.dstore_keys[:] = 1
        datastore.dstore_vals[:] = 1

        keys_path = datastore._keys_path
        keys = datastore.dstore_keys.copy()

        del datastore

        # Check if values were saved in disk
        loaded = np.memmap(keys_path, mode="r", dtype=np.float16, shape=(size, 10))
        assert (loaded == keys).all()
        assert (loaded == np.ones((size, 10))).all()

        new_size = 500
        datastore = Datastore(
            dstore_dir=dstore_dir,
            dimension=10,
            model_type="model",
            device="cuda",
            flat_index=False,
            continue_writing=True,
            dstore_size=new_size,
        ).load_keys_and_vals()
        datastore.dstore_keys[size:] = 2

        # Check file existance
        dstore_dir = Path(dstore_dir)
        total_size = size + new_size
        assert datastore.dstore_size == total_size, "Datastore size should be of 150."
        assert (dstore_dir / f"dstore_model_{total_size}_10_keys.npy").exists(), (
            f"Datastore file with {total_size} tokens not found. "
            "There may be something wrong with the continual learning strategy."
        )
        assert not (
            dstore_dir / f"dstore_model_{size}_10_keys.npy"
        ).exists(), f"Datastore file with {size} tokens should not exist."
        assert not (
            dstore_dir / f"dstore_model_{new_size}_10_keys.npy"
        ).exists(), f"Datastore file with {new_size} tokens should not exist."

        # Check value concatenation
        assert datastore.dstore_keys[0, 0] == 1
        assert datastore.dstore_keys[size, 0] == 2
        assert datastore.dstore_vals[0] == 1
        assert datastore.dstore_vals[size] == 0

        del datastore

        # Test train
        datastore = Datastore(
            dstore_dir=dstore_dir,
            dimension=10,
            model_type="model",
            device="cuda",
            flat_index=False,
            continue_writing=False,  # Should be false when training index
            dstore_size=None,
        ).load_keys_and_vals()
        datastore.build_index(
            num_keys_to_add_at_a_time=1000000,
            ncentroids=1,
            seed=1,
            code_size=1,
            probe=1,
        )

        assert datastore.dstore_size == total_size, "Datastore size should be of 150."
        assert (
            dstore_dir / f"index_model_{total_size}_10.indexed"
        ).exists(), "Trained index not found."

        # When training the index, if we set "continue_writing" to true
        # and dstore_size != None, we'll increase the datastore size
        # That's not the desired behavior so it raises a warning.
        # Either continue_writing has to be False, or dstore_size = None
        datastore = Datastore(
            dstore_dir=dstore_dir,
            dimension=10,
            model_type="model",
            device="cuda",
            flat_index=False,
            continue_writing=True,
            dstore_size=100,
        ).load_keys_and_vals()

        with pytest.raises(RuntimeError):
            datastore.build_index()


def test_files_removal():
    """Test if previous index and dstore files are removed correctly."""
    with tempfile.TemporaryDirectory() as dstore_dir:
        dstore_dir = Path(dstore_dir)
        size = 1000
        datastore = Datastore(
            dstore_dir=dstore_dir,
            dimension=10,
            model_type="model",
            device="cuda",
            flat_index=False,
            continue_writing=True,
            dstore_size=size,
        ).load_keys_and_vals()
        datastore.dstore_keys[:] = 1
        datastore.build_index(
            num_keys_to_add_at_a_time=1000000,
            ncentroids=1,
            seed=1,
            code_size=1,
            probe=1,
        )
        del datastore

        dstores = list(dstore_dir.glob("dstore_*"))
        assert len(dstores) == 2, "More than two files starting with `dstore` found."

        size = 500
        datastore = Datastore(
            dstore_dir=dstore_dir,
            dimension=10,
            model_type="model",
            device="cuda",
            flat_index=False,
            continue_writing=True,
            dstore_size=size,
        ).load_keys_and_vals()
        datastore.dstore_keys[:] = 2
        datastore.continue_writing = False

        dstores = list(dstore_dir.glob("dstore_*"))
        assert len(dstores) == 2, "More than two files starting with `dstore` found."
        indexes = list(dstore_dir.glob("index_*"))
        assert (
            len(indexes) == 0
        ), "Index file from the previous datastore was not deleted."
