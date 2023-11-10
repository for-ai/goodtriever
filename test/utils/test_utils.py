from pathlib import Path

import pytest

from utils.utils import structure_output_filepath


def test_structure_output_filepath():
    """Test automatic filepath structuring"""
    output_folder = Path("teste/")
    filename = "eos_gpt2"

    # Test generation step
    filename = structure_output_filepath(
        step="generation",
        output_folder=output_folder,
        previous_filename=filename,
        mkdir=False,
    )
    assert filename == output_folder / "eos_gpt2_generations.jsonl"

    # Test perspective step
    filename = structure_output_filepath(
        step="perspective",
        output_folder=filename.parent,
        previous_filename=filename.name,  # Test if can handle string filename
        mkdir=False,
    )
    assert filename == output_folder / "eos_gpt2_perspective.jsonl"

    # Test random filename (does not contain `generations` string)
    random_filename = structure_output_filepath(
        step="perspective",
        output_folder=None,  # Test if can handle None output_folder
        previous_filename=output_folder / "random_filename",
        mkdir=False,
    )
    assert random_filename == output_folder / "random_filename_perspective.jsonl"

    # Test collate step
    filename = structure_output_filepath(
        step="collate",
        output_folder=Path("different_output/"),
        previous_filename=filename.stem,  # Test if can handle stem string
        mkdir=False,
    )
    assert filename == Path("different_output/") / "eos_gpt2_collated.jsonl"

    # Test error raise with invalid step
    with pytest.raises(NotImplementedError):
        _ = structure_output_filepath(
            step="random_step",
            output_folder=None,
            previous_filename=filename,
            mkdir=False,
        )
