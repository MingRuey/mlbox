import pathlib

SAMPLE_DIR = pathlib.Path(__file__).parent.joinpath("samples")
assert SAMPLE_DIR.is_dir()
