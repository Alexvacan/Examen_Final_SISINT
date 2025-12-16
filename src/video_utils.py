import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple


_FRAME_TIME_RE = re.compile(r"_t([0-9]+(?:\.[0-9]+)?)\.jpg$", re.IGNORECASE)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str, indent: int = 2) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def list_subdirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    out = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    out.sort()
    return out


def parse_frame_time_from_name(filename: str) -> Optional[float]:
    """
    Espera nombres tipo: frame_000056_t3.38.jpg
    """
    m = _FRAME_TIME_RE.search(filename)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_ts(value: Any) -> Optional[float]:
    """
    Normaliza timestamps a float o None.
    """
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def segment_contains_t(seg: Dict[str, Any], t: float) -> bool:
    s = normalize_ts(seg.get("start"))
    e = normalize_ts(seg.get("end"))
    if s is None or e is None:
        return False
    return s <= t <= e


def safe_basename_no_ext(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]
