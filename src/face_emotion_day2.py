import os
import re
import json
from typing import Dict, List, Optional, Any

import cv2
from deepface import DeepFace


_TIME_RE = re.compile(r"_t([0-9]+(?:\.[0-9]+)?)\.jpg$", re.IGNORECASE)


def _frame_time_from_name(filename: str) -> Optional[float]:
    m = _TIME_RE.search(filename)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _enhance_clahe_bgr(img_bgr):
    # mejora iluminación de forma barata y estable
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def _to_float_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (d or {}).items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out


def analyze_frames_dir(
    frames_dir: str,
    enhance: bool = True,
    enforce_detection: bool = False,
) -> Dict[str, Any]:
    """
    Procesa TODOS los .jpg en frames_dir y devuelve una serie temporal:
    items: [{t, frame, dominant_emotion, scores}] + errores por frame si aplica
    """
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"No existe la carpeta: {frames_dir}")

    frames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])
    if not frames:
        return {
            "frames_dir": frames_dir,
            "n_frames": 0,
            "items": [],
            "errors": ["No se encontraron .jpg en la carpeta"]
        }

    items: List[Dict[str, Any]] = []

    for fname in frames:
        fpath = os.path.join(frames_dir, fname)
        t = _frame_time_from_name(fname)  # preferido (porque ya lo tienes en el nombre)

        try:
            img = cv2.imread(fpath)
            if img is None:
                raise ValueError("Imagen no pudo cargarse (cv2.imread devolvió None)")

            if enhance:
                img = _enhance_clahe_bgr(img)

            r = DeepFace.analyze(
                img_path=img,
                actions=["emotion"],
                enforce_detection=enforce_detection,
            )
            if isinstance(r, list):
                r = r[0]

            dom = r.get("dominant_emotion")
            scores = _to_float_dict(r.get("emotion", {}))

            items.append({
                "t": t,
                "frame": fname,
                "dominant_emotion": dom,
                "scores": scores
            })

        except Exception as e:
            # No se cae el pipeline: registra error y continúa
            items.append({
                "t": t,
                "frame": fname,
                "error": str(e)
            })

    # ordenar por tiempo si existe; si t es None, queda al final por frame name
    items.sort(key=lambda x: (x["t"] is None, x["t"] if x["t"] is not None else 0.0, x["frame"]))

    return {
        "frames_dir": frames_dir,
        "n_frames": len(frames),
        "items": items
    }


def save_json(data: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
