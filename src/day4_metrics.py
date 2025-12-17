import json
from typing import Dict, Any, Optional, List, Tuple
from video_utils import normalize_ts


def _safe_str(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _extract_face(item: Dict[str, Any]) -> Optional[str]:
    return _safe_str((item.get("face") or {}).get("dominant"))


def _extract_text(item: Dict[str, Any]) -> Optional[str]:
    return _safe_str((item.get("text") or {}).get("dominant"))


def _load_labels(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _label_at_t(labels: Dict[str, Any], t: float) -> Optional[str]:
    """
    labels format:
      { segments: [{start,end,emotion}, ...] }
    """
    segs = labels.get("segments", [])
    for s in segs:
        st = normalize_ts(s.get("start"))
        en = normalize_ts(s.get("end"))
        emo = _safe_str(s.get("emotion"))
        if st is None or en is None or emo is None:
            continue
        if st <= t <= en:
            return emo
    return None


def congruence_face_vs_text(multimodal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Congruencia instantánea: face == text cuando text_emotion existe.
    """
    items = multimodal.get("items", [])
    total_with_text = 0
    match = 0
    mismatch = 0

    for it in items:
        face = _extract_face(it)
        text = _extract_text(it)
        if not face:
            continue
        if text is None:
            continue

        total_with_text += 1
        if face == text:
            match += 1
        else:
            mismatch += 1

    rate = (match / total_with_text) if total_with_text > 0 else None

    return {
        "total_with_text": total_with_text,
        "match": match,
        "mismatch": mismatch,
        "match_rate": rate
    }


def congruence_vs_manual_labels(multimodal: Dict[str, Any], labels_path: str) -> Dict[str, Any]:
    """
    Compara emoción facial vs etiqueta manual por tiempo t.
    (Puedes cambiar a text si quieres, pero facial es lo más estable).
    """
    labels = _load_labels(labels_path)

    items = multimodal.get("items", [])
    total_labeled = 0
    match = 0
    mismatch = 0
    unknown = 0  # frames que no caen en ningún segmento etiquetado

    for it in items:
        t = normalize_ts(it.get("t"))
        if t is None:
            continue
        face = _extract_face(it)
        if not face:
            continue

        gt = _label_at_t(labels, t)
        if gt is None:
            unknown += 1
            continue

        total_labeled += 1
        if face == gt:
            match += 1
        else:
            mismatch += 1

    rate = (match / total_labeled) if total_labeled > 0 else None

    return {
        "labels_path": labels_path,
        "total_labeled": total_labeled,
        "match": match,
        "mismatch": mismatch,
        "match_rate": rate,
        "unknown_frames": unknown
    }
