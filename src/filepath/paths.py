from pathlib import Path

_FILEPATH_DIR = Path(__file__).resolve().parent
_SRC_DIR = _FILEPATH_DIR.parent
PROJECT_ROOT = _SRC_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
SRC_DIR = _SRC_DIR
