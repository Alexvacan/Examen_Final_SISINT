from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

from video_utils import normalize_ts, read_json

def _majority_vote(seq: List[str]) -> str:
    c = Counter(seq)
    # desempate determinista: el que aparece primero en la ventana
    top = c.most_common()
    best_count = top[0][1]
    candidates = {k for k,v in top if v == best_count}
    for x in seq:
        if x in candidates:
            return x
    return top[0][0]

def smooth_sequence(values: List[Optional[str]], k: int = 5) -> List[Optional[str]]:
    if k <= 1:
        return values[:]
    n = len(values)
    half = k // 2
    out: List[Optional[str]] = [None]*n
    for i in range(n):
        window = [values[j] for j in range(max(0,i-half), min(n,i+half+1)) if values[j] is not None]
        out[i] = _majority_vote(window) if window else None
    return out

# ----------------------------
# Helpers
# ----------------------------
def _safe_str(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None

def _build_text_segments(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Construye segmentos de texto continuos:
    [{start, end, label}]
    """
    segments = []
    current = None

    for it in items:
        t = normalize_ts(it.get("t"))
        if t is None:
            continue

        txt = it.get("text")

        # --- FIX CLAVE: text puede ser dict o string ---
        if isinstance(txt, dict):
            txt = txt.get("raw") or txt.get("text") or ""
        elif not isinstance(txt, str):
            txt = ""

        txt = txt.strip()
        if not txt:
            continue

        label = _map_dom(txt, TEXT_MAP)
        if label is None:
            continue

        if current is None:
            current = {
                "start": float(t),
                "end": float(t),
                "label": label
            }
        elif current["label"] == label:
            current["end"] = float(t)
        else:
            segments.append(current)
            current = {
                "start": float(t),
                "end": float(t),
                "label": label
            }

    if current:
        segments.append(current)

    return segments

def _text_dom_for_time(t: float, segments: List[Dict[str, Any]]) -> Optional[str]:
    """
    Devuelve la emoción de texto correspondiente al tiempo t
    usando segmentos [{start,end,label}]
    """
    if t is None:
        return None

    for s in segments:
        try:
            if float(s["start"]) <= t <= float(s["end"]):
                return s.get("label")
        except Exception:
            continue

    return None

def _extract_text_dom_direct(it: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extrae emoción dominante desde texto SIN segmentación.
    Retorna: (dominant, dominant_raw, source)
    """
    txt = it.get("text")

    # text puede ser dict o string
    if isinstance(txt, dict):
        raw = txt.get("raw") or txt.get("text")
    else:
        raw = txt

    raw = _safe_str(raw)
    if raw is None:
        return None, None, None

    dom = _map_dom(raw, TEXT_MAP)
    return dom, raw, "text"

# ----------------------------
# Taxonomías y mapeos
# ----------------------------
# Face model (angry/disgust/fear/happy/sad/surprise/neutral)
FACE_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

# Text model (joy/anger/sadness/others, a veces fear/disgust/surprise)
TEXT_MAP = {
    "joy": "happy",
    "anger": "angry",
    "sadness": "sad",
    "surprise": "surprise",
    "fear": "fear",
    "disgust": "disgust",
    "others": "neutral",  # <- clave: ya no se descarta
    "neutral": "neutral",
    # por si tu modelo devuelve otras variantes:
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
}

# Manual labels (ej típico en español)
MANUAL_MAP = {
    "enojo": "angry",
    "enojado": "angry",
    "ira": "angry",
    "anger": "angry",

    "feliz": "happy",
    "alegria": "happy",
    "alegría": "happy",
    "joy": "happy",
    "happy": "happy",

    "triste": "sad",
    "tristeza": "sad",
    "sad": "sad",
    "sadness": "sad",

    "miedo": "fear",
    "fear": "fear",

    "asco": "disgust",
    "disgust": "disgust",

    "sorpresa": "surprise",
    "surprise": "surprise",

    "neutral": "neutral",
    "serio": "neutral",
    "normal": "neutral",
    "others": "neutral",
}


def _map_dom(dom: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
    if dom is None:
        return None
    k = dom.strip().lower()
    return mapping.get(k)


def _extract_face_dom(item: Dict[str, Any]) -> Optional[str]:
    face = item.get("face") or {}

    # Dominante original del modelo
    dom_raw = _safe_str(face.get("dominant"))
    dom_adjusted = dom_raw

    scores = face.get("scores") or {}

    if USE_FACE_HEURISTIC and dom_raw:
        try:
            angry = float(scores.get("angry", 0))
            fear = float(scores.get("fear", 0))
            disgust = float(scores.get("disgust", 0))

            # Heurística: enojo camuflado como fear
            if (
                dom_raw == "fear"
                and fear > ANGER_FEAR_THRESHOLD
                and angry > ANGER_MIN_SCORE
                and disgust > DISGUST_MIN_SCORE
            ):
                dom_adjusted = "angry"

        except Exception:
            pass

    # Guardamos ambos (muy útil para análisis posterior)
    face["dominant_raw"] = dom_raw
    face["dominant_adjusted"] = dom_adjusted

    return _map_dom(dom_adjusted, FACE_MAP)

    # ----------------------------
# Configuración heurística facial
# ----------------------------
# --- arriba del archivo, pon esto como constantes ---
# ----------------------------
# Heurística ON/OFF + umbrales
# ----------------------------
HEURISTIC_ON = True
TH_FEAR = 50.0
TH_ANGRY = 5.0
TH_DISGUST = 0.3


def _extract_face_dom(item: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    """
    Retorna:
      (dominant_mapeado, was_adjusted)
    Además, si hay scores, guarda:
      face['dominant_raw'], face['dominant_adjusted']
    """
    face = item.get("face") or {}
    dom_raw = _safe_str(face.get("dominant"))
    scores = face.get("scores") or {}

    adjusted = False
    dom_adj = dom_raw

    if HEURISTIC_ON and scores:
        try:
            angry = float(scores.get("angry", 0))
            fear = float(scores.get("fear", 0))
            disgust = float(scores.get("disgust", 0))

            # Heurística: "enojo camuflado como fear"
            if (fear > TH_FEAR) and (angry > TH_ANGRY) and (disgust > TH_DISGUST):
                if dom_adj != "angry":
                    dom_adj = "angry"
                    adjusted = True
        except Exception:
            pass

    # Guardamos debug en el propio item (sirve para inspeccionar luego)
    face["dominant_raw"] = dom_raw
    face["dominant_adjusted"] = dom_adj
    item["face"] = face

    return _map_dom(dom_adj, FACE_MAP), adjusted

# ----------------------------
# Métrica: congruencia cara vs texto
# ----------------------------
def congruence_face_vs_text(multimodal: Dict[str, Any]) -> Dict[str, Any]:
    items = multimodal.get("items", []) or []
    segs = _build_text_segments(items)

    total_with_text = 0
    match = 0
    mismatch = 0
    skipped_no_face = 0
    skipped_no_text = 0
    skipped_unmappable_face = 0
    skipped_unmappable_text = 0

    for it in items:
        t = normalize_ts(it.get("t"))
        if t is None:
            continue
        t = float(t)

        face_dom, _ = _extract_face_dom(it)
        if face_dom is None:
            skipped_no_face += 1
            continue
        if face_dom not in FACE_MAP.values():
            skipped_unmappable_face += 1
            continue

        text_dom_direct, _, _ = _extract_text_dom_direct(it)
        text_dom = text_dom_direct if text_dom_direct else _text_dom_for_time(t, segs)

        if text_dom is None:
            skipped_no_text += 1
            continue
        if text_dom not in FACE_MAP.values():
            skipped_unmappable_text += 1
            continue

        total_with_text += 1
        if face_dom == text_dom:
            match += 1
        else:
            mismatch += 1

    match_rate = (match / total_with_text) if total_with_text > 0 else None

    return {
        "total_with_text": total_with_text,
        "match": match,
        "mismatch": mismatch,
        "match_rate": match_rate,
        "skipped_no_face": skipped_no_face,
        "skipped_no_text": skipped_no_text,
        "skipped_unmappable_face": skipped_unmappable_face,
        "skipped_unmappable_text": skipped_unmappable_text,
        "text_segments": len(segs),
    }


# ----------------------------
# Carga de labels manuales (robusta)
# ----------------------------
def _load_labels(labels_path: str) -> List[Dict[str, Any]]:
    """
    Devuelve labels normalizados como lista de dicts.
    Soporta:
      - list directo
      - dict con items/labels/frames/...
      - mapping frame->label
      - TU FORMATO: {"segments":[{"start":..,"end":..,"emotion":..}, ...]}
    """
    d = read_json(labels_path)

    # A) LISTA directa
    if isinstance(d, list):
        if d and isinstance(d[0], str):
            return [{"frame": i, "label": d[i]} for i in range(len(d))]
        return d

    # B) DICT
    if isinstance(d, dict):
        # B1) Formato por segmentos (TU CASO)
        if isinstance(d.get("segments"), list):
            segs = d["segments"]
            out = []
            for s in segs:
                if not isinstance(s, dict):
                    continue
                start = normalize_ts(s.get("start"))
                end = normalize_ts(s.get("end"))
                emo = s.get("emotion") or s.get("label")
                if start is None or end is None or emo is None:
                    continue
                out.append({"start": float(start), "end": float(end), "label": str(emo).strip()})
            if out:
                return [{"segments": out}]

        # B2) diccionario con lista adentro
        for key in ["items", "labels", "frames", "annotations", "data", "results"]:
            v = d.get(key)
            if isinstance(v, list):
                if v and isinstance(v[0], str):
                    return [{"frame": i, "label": v[i]} for i in range(len(v))]
                return v

        # B3) mapping frame->label
        if d:
            vals = list(d.values())
            str_vals = [x for x in vals if isinstance(x, str)]
            if len(str_vals) >= max(1, len(vals)//2):
                out = []
                for k, v in d.items():
                    if not isinstance(v, str):
                        continue
                    kk = str(k)
                    digits = "".join(ch for ch in kk if ch.isdigit())
                    if digits:
                        out.append({"frame": int(digits), "label": v})
                if out:
                    out.sort(key=lambda x: x.get("frame", 0))
                    return out

    raise ValueError(f"Formato de labels no reconocido en {labels_path}")


def _extract_manual_label(x: Dict[str, Any]) -> Optional[str]:
    for k in ["label", "emotion", "value", "dominant"]:
        v = _safe_str(x.get(k))
        if v:
            return v
    return None


def _extract_manual_time(x: Dict[str, Any]) -> Optional[float]:
    for k in ["t", "time", "timestamp", "seconds"]:
        v = normalize_ts(x.get(k))
        if v is not None:
            return float(v)
    return None

def congruence_vs_manual_labels(multimodal: Dict[str, Any], labels_path: str) -> Dict[str, Any]:
    items = multimodal.get("items", []) or []
    n_adjusted = 0

    labels = _load_labels(labels_path)

    # ✅ FIX CLAVE: si viene como [{"segments": [...]}], lo expandimos a [{t,label},...]
    if labels and isinstance(labels[0], dict) and "segments" in labels[0]:
        segments = labels[0]["segments"]  # [{start,end,label}, ...]

        def label_for_t(t: float) -> Optional[str]:
            for s in segments:
                if float(s["start"]) <= t <= float(s["end"]):
                    return s["label"]
            return None

        expanded: List[Dict[str, Any]] = []
        for it in items:
            t = normalize_ts(it.get("t"))
            if t is None:
                continue
            lab = label_for_t(float(t))
            if lab is not None:
                expanded.append({"t": float(t), "label": lab})

        labels = expanded

    total_labeled = 0
    match = 0
    mismatch = 0
    unknown_frames = 0
    skipped_no_pred = 0

    # ---------------------------------------------------------
    # ✅ Construimos (t -> face_dom) y APLICAMOS SUAVIZADO
    # ---------------------------------------------------------
    ts: List[float] = []
    raw_face: List[Optional[str]] = []

    for it in items:
        t = normalize_ts(it.get("t"))
        if t is None:
            continue

        dom, was_adj = _extract_face_dom(it)
        if was_adj:
            n_adjusted += 1

        ts.append(float(t))
        raw_face.append(dom)  # SOLO el dominant (string)

    # Suavizado temporal
    sm_face = smooth_sequence(raw_face, k=5)

    frames: List[Tuple[float, str]] = []
    for t, dom in zip(ts, sm_face):
        if dom:
            frames.append((t, dom))
    frames.sort(key=lambda x: x[0])

    def nearest_face(t: float) -> Optional[str]:
        if not frames:
            return None
        best = None
        best_dt = None
        for ft, fdom in frames:
            dt = abs(ft - t)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = fdom
        return best

    compared_signal = "face_smoothed"

    # ---------------------------------------------------------
    # ✅ Comparación contra labels manuales
    # ---------------------------------------------------------
    for lb in labels:
        if not isinstance(lb, dict):
            continue

        raw = _extract_manual_label(lb)
        gt = _map_dom(raw, MANUAL_MAP)
        if gt is None:
            unknown_frames += 1
            continue

        t = _extract_manual_time(lb)

        if t is None:
            idx = lb.get("frame") or lb.get("frame_idx") or lb.get("idx")
            try:
                idx = int(idx)
            except Exception:
                unknown_frames += 1
                continue

            if 0 <= idx < len(items):
                try:
                    pred = sm_face[idx] if idx < len(sm_face) else _extract_face_dom(items[idx])[0]
                except Exception:
                    pred = _extract_face_dom(items[idx])[0]
            else:
                pred = None
        else:
            pred = nearest_face(float(t))

        if pred is None:
            skipped_no_pred += 1
            continue

        total_labeled += 1
        if pred == gt:
            match += 1
        else:
            mismatch += 1

    match_rate = (match / total_labeled) if total_labeled > 0 else None

    return {
        "labels_path": labels_path,
        "gt_taxonomy": "normalized_manual_map",
        "compared_signal": compared_signal,
        "total_labeled": total_labeled,
        "match": match,
        "mismatch": mismatch,
        "match_rate": match_rate,
        "unknown_frames": unknown_frames,
        "skipped_no_pred": skipped_no_pred,

        # --- extras debug/heurística ---
        "n_adjusted": n_adjusted,
        "heuristic_on": HEURISTIC_ON,
        "thresholds": {
            "fear": TH_FEAR,
            "angry": TH_ANGRY,
            "disgust": TH_DISGUST
        }
    }
