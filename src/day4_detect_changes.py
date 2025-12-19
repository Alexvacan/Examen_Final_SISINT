from typing import Dict, Any, List, Optional, Tuple
from video_utils import normalize_ts


# -----------------------------
# Helpers
# -----------------------------
def _safe_str(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    return s if s else None


def _extract_face_emotion(item: Dict[str, Any]) -> Optional[str]:
    face = item.get("face") or {}
    return _safe_str(face.get("dominant"))


def _extract_text_emotion(item: Dict[str, Any]) -> Optional[str]:
    """
    Extrae emoción de texto.
    ⚠️ 'others' NO se considera emoción válida.
    """
    text = item.get("text") or {}
    dom = _safe_str(text.get("dominant"))

    if dom in (None, "others", "other"):
        return None

    return dom


# -----------------------------
# Detect changes
# -----------------------------
def detect_changes(multimodal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detecta cambios en emoción facial y emoción de texto a lo largo del tiempo.
    Retorna eventos con:
      - t
      - source: 'face' | 'text'
      - from / to
    """
    items = multimodal.get("items", [])

    # ordenar por tiempo
    rows: List[Tuple[float, Dict[str, Any]]] = []
    for it in items:
        t = normalize_ts(it.get("t"))
        if t is not None:
            rows.append((t, it))

    rows.sort(key=lambda x: x[0])

    face_events: List[Dict[str, Any]] = []
    text_events: List[Dict[str, Any]] = []

    prev_face: Optional[str] = None
    prev_text: Optional[str] = None

    for t, it in rows:
        face = _extract_face_emotion(it)
        text = _extract_text_emotion(it)

        # ---- FACE ----
        if face:
            if prev_face and face != prev_face:
                face_events.append({
                    "t": t,
                    "source": "face",
                    "from": prev_face,
                    "to": face
                })
            prev_face = face

        # ---- TEXT ----
        # solo contamos cambios si hay emoción válida
        if text:
            if prev_text and text != prev_text:
                text_events.append({
                    "t": t,
                    "source": "text",
                    "from": prev_text,
                    "to": text
                })
            prev_text = text

    return {
        "n_face_changes": len(face_events),
        "n_text_changes": len(text_events),
        "face_changes": face_events,
        "text_changes": text_events
    }
