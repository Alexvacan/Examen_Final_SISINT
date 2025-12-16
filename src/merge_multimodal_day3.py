import os
from typing import Dict, Any, List, Optional

from video_utils import read_json, write_json, normalize_ts
from logger_utils import get_logger

log = get_logger("merge_day3")


def _coalesce_text(item: Dict[str, Any]) -> Optional[str]:
    # texto puede venir None si no cayó en ningún segmento
    txt = item.get("text")
    if txt is None:
        return None
    txt = str(txt).strip()
    return txt if txt else None


def build_multimodal_from_sync(sync_data: Dict[str, Any], video_name: str) -> Dict[str, Any]:
    """
    Convierte outputs/sync_preview/<name>_sync_preview.json a un JSON multimodal final.
    - 1 item por frame (tiempo t)
    - incluye emoción facial + emoción texto (si existe) + texto
    """

    items = sync_data.get("items", [])
    out_items: List[Dict[str, Any]] = []

    for x in items:
        t = normalize_ts(x.get("t"))
        if t is None:
            continue

        face_emotion = x.get("face_emotion")
        face_scores = x.get("face_scores")

        text = _coalesce_text(x)

        text_emotion = x.get("text_emotion")
        text_scores = x.get("text_scores")

        out_items.append({
            "t": t,
            "frame": x.get("frame"),

            "face": {
                "dominant": face_emotion,
                "scores": face_scores
            },

            "text": {
                "segment_start": normalize_ts(x.get("text_start")),
                "segment_end": normalize_ts(x.get("text_end")),
                "content": text,
                "dominant": text_emotion,
                "scores": text_scores
            }
        })

    out_items.sort(key=lambda r: r["t"])

    return {
        "video": video_name,
        "n_items": len(out_items),
        "items": out_items
    }


def main():
    """
    Runner: lee el sync_preview y genera el multimodal final en outputs/multimodal/
    """
    import argparse

    ap = argparse.ArgumentParser(description="Día 3 Paso 2: construir JSON multimodal final")
    ap.add_argument("--name", required=True, help="nombre base del video (ej: prueba1)")
    ap.add_argument("--sync-dir", default="outputs/sync_preview", help="carpeta sync_preview")
    ap.add_argument("--out-dir", default="outputs/multimodal", help="carpeta salida multimodal")
    args = ap.parse_args()

    name = args.name
    sync_path = os.path.join(args.sync_dir, f"{name}_sync_preview.json")

    if not os.path.exists(sync_path):
        raise SystemExit(f"Falta el sync preview. Ejecuta primero Paso 1. No existe: {sync_path}")

    sync_data = read_json(sync_path)
    multimodal = build_multimodal_from_sync(sync_data, video_name=name)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{name}_multimodal.json")
    write_json(multimodal, out_path)

    log.info(f"✅ Multimodal guardado: {out_path}")
    log.info(f"n_items={multimodal['n_items']}")
    if multimodal["items"]:
        log.info(f"Ejemplo: {multimodal['items'][0]}")


if __name__ == "__main__":
    main()
