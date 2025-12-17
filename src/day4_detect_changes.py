from typing import Dict, Any, List, Optional, Tuple
from video_utils import normalize_ts


def _safe_str(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _extract_face_emotion(item: Dict[str, Any]) -> Optional[str]:
    face = item.get("face") or {}
    return _safe_str(face.get("dominant"))


def _extract_text_emotion(item: Dict[str, Any]) -> Optional[str]:
    text = item.get("text") or {}
    return _safe_str(text.get("dominant"))


def detect_changes(multimodal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detecta cambios en emoción facial y emoción de texto a lo largo del tiempo.
    Retorna eventos con:
      - t
      - source: 'face' | 'text'
      - from / to
    """
    items = multimodal.get("items", [])
    # ordenar por t
    rows: List[Tuple[float, Dict[str, Any]]] = []
    for it in items:
        t = normalize_ts(it.get("t"))
        if t is None:
            continue
        rows.append((t, it))
    rows.sort(key=lambda x: x[0])

    face_events: List[Dict[str, Any]] = []
    text_events: List[Dict[str, Any]] = []

    prev_face = None
    prev_text = None

    for t, it in rows:
        face = _extract_face_emotion(it)
        text = _extract_text_emotion(it)

        if face and prev_face and face != prev_face:
            face_events.append({"t": t, "source": "face", "from": prev_face, "to": face})
        if face and prev_face is None:
            prev_face = face
        elif face:
            prev_face = face

        # text puede ser None (si no hay emoción por texto); solo contamos cambios cuando existe
        if text and prev_text and text != prev_text:
            text_events.append({"t": t, "source": "text", "from": prev_text, "to": text})
        if text and prev_text is None:
            prev_text = text
        elif text:
            prev_text = text

    return {
        "n_face_changes": len(face_events),
        "n_text_changes": len(text_events),
        "face_changes": face_events,
        "text_changes": text_events
    }
