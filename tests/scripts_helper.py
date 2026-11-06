import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_synthetic_data", Path(__file__).parent.parent / "scripts" / "make_synthetic_data.py"
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
synthetic_frame = _mod.synthetic_frame
