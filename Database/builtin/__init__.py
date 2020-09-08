from pathlib import Path

from .config import DatasetConfig


_JSONS = [str(file) for file in Path(__file__).parent.glob("*.json")]

BUILT_INS = [
    DatasetConfig.from_json(json) for json in _JSONS
]
