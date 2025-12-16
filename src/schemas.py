from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


# ---------- A: Face emotion timeseries ----------

@dataclass
class FaceEmotionItem:
    t: Optional[float]
    frame: str
    dominant_emotion: Optional[str] = None
    scores: Optional[Dict[str, float]] = None
    error: Optional[str] = None


@dataclass
class FaceEmotionSeries:
    frames_dir: str
    n_frames: int
    items: List[Dict[str, Any]]  # guardamos items ya "serializados"


# ---------- B: Text emotion per segment ----------

@dataclass
class TextEmotionItem:
    start: float
    end: float
    text: str
    dominant_emotion: Optional[str] = None
    scores: Optional[Dict[str, float]] = None
    error: Optional[str] = None


@dataclass
class TextEmotionSeries:
    n_segments: int
    items: List[Dict[str, Any]]


# ---------- Helpers para asegurar formato ----------

def to_dict(obj) -> Dict[str, Any]:
    return asdict(obj)
