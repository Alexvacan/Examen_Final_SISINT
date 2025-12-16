import os
from typing import Dict, Any, List, Optional

from video_utils import read_json, write_json, normalize_ts
from logger_utils import get_logger

log = get_logger("sync_day3")


def _find_segment(segments: List[Dict[str, Any]], t: float) -> Optional[Dict[str, Any]]:
    """
    Retorna el primer segmento donde start <= t <= end.
    """
    for s in segments:
        start = normalize_ts(s.get("start"))
        end = normalize_ts(s.get("end"))
        if start is None or end is None:
            continue
        if start <= t <= end:
            return s
    return None


def sync_face_with_text_segments(
    face_timeseries: Dict[str, Any],
    transcript: Dict[str, Any],
    text_emotions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Une por tiempo:
      - face_timeseries.items[].t
      - transcript.segments[] (start/end/text)
      - (opcional) text_emotions.items[] (start/end/dominant_emotion/scores)
    Devuelve un dict con items sincronizados (uno por frame válido).
    """

    face_items = face_timeseries.get("items", [])
    segments = transcript.get("segments", [])

    text_items = None
    if text_emotions is not None:
        text_items = text_emotions.get("items", [])

    synced: List[Dict[str, Any]] = []
    dropped_no_t = 0
    dropped_no_face = 0

    for f in face_items:
        t = normalize_ts(f.get("t"))
        if t is None:
            dropped_no_t += 1
            continue

        # Si el frame tuvo error o no tiene emoción, lo saltamos (evita basura en el merge)
        if f.get("error") is not None:
            continue
        if not f.get("dominant_emotion"):
            dropped_no_face += 1
            continue

        seg = _find_segment(segments, t)
        txt = None
        seg_start = None
        seg_end = None
        if seg:
            txt = seg.get("text")
            seg_start = normalize_ts(seg.get("start"))
            seg_end = normalize_ts(seg.get("end"))

        # Si hay emociones de texto, buscamos el item correspondiente
        txt_emotion = None
        txt_scores = None
        if text_items is not None and seg_start is not None and seg_end is not None:
            # busca item de texto cuyo start/end coincidan (o contengan) el frame
            # (más robusto que match exacto)
            for ti in text_items:
                ts = normalize_ts(ti.get("start"))
                te = normalize_ts(ti.get("end"))
                if ts is None or te is None:
                    continue
                if ts <= t <= te:
                    txt_emotion = ti.get("dominant_emotion")
                    txt_scores = ti.get("scores")
                    break

        synced.append({
            "t": t,
            "frame": f.get("frame"),
            "face_emotion": f.get("dominant_emotion"),
            "face_scores": f.get("scores"),
            "text": txt,
            "text_start": seg_start,
            "text_end": seg_end,
            "text_emotion": txt_emotion,
            "text_scores": txt_scores,
        })

    synced.sort(key=lambda x: x["t"])

    return {
        "n_synced": len(synced),
        "dropped_no_t": dropped_no_t,
        "dropped_no_face": dropped_no_face,
        "items": synced
    }


def main():
    """
    Runner simple para probar Paso 1 con un video.
    Genera un archivo temporal en outputs/sync_preview/
    """
    import argparse

    ap = argparse.ArgumentParser(description="Día 3 Paso 1: sincronizar timestamps")
    ap.add_argument("--name", required=True, help="nombre base del video (ej: prueba1)")
    ap.add_argument("--face", default="outputs/face_emotions", help="carpeta face_emotions")
    ap.add_argument("--tr", default="outputs/transcripts", help="carpeta transcripts")
    ap.add_argument("--txt", default="outputs/text_emotions", help="carpeta text_emotions")
    ap.add_argument("--out", default="outputs/sync_preview", help="salida preview")
    args = ap.parse_args()

    name = args.name

    face_path = os.path.join(args.face, f"{name}_face_timeseries.json")
    tr_path = os.path.join(args.tr, f"{name}_transcript.json")
    txt_path = os.path.join(args.txt, f"{name}_text_emotions.json")

    if not os.path.exists(face_path):
        raise SystemExit(f"Falta: {face_path}")
    if not os.path.exists(tr_path):
        raise SystemExit(f"Falta: {tr_path}")

    face = read_json(face_path)
    tr = read_json(tr_path)

    text_emotions = None
    if os.path.exists(txt_path):
        text_emotions = read_json(txt_path)
    else:
        log.info("No existe text_emotions, se sincronizará solo con transcript.")

    data = sync_face_with_text_segments(face, tr, text_emotions)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{name}_sync_preview.json")
    write_json(data, out_path)

    log.info(f"✅ Preview guardado: {out_path}")
    log.info(f"n_synced={data['n_synced']} | dropped_no_t={data['dropped_no_t']} | dropped_no_face={data['dropped_no_face']}")
    if data["items"]:
        log.info(f"Ejemplo: {data['items'][0]}")


if __name__ == "__main__":
    main()
