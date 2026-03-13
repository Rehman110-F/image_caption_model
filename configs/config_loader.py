import yaml
from pathlib import Path
from types import SimpleNamespace

def _to_ns(d):                          # helper: converts dict → object
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _to_ns(v) if isinstance(v, dict) else v)
    return ns

def load_config(path=None):
    if path is None:
        path = Path(__file__).parent / "config.yaml"   # default path
    with open(path) as f:
        return _to_ns(yaml.safe_load(f))               # read YAML → return object
    
CFG = load_config() 