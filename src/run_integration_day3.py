import os

from video_utils import list_subdirs, read_json, write_json
from logger_utils import get_logger

from sync_timestamps_day3 import sync_face_with_text_segments
from merge_multimodal_day3 import build_multimodal_from_sync

log = get_logger("run_day3")


def process_one(name: str,
                face_dir: str,
                tr_dir: str,
                txt_dir: str,
                out_dir: str) -> str:
    face_path = os.path.join(face_dir, f"{name}_face_timeseries.json")
    tr_path = os.path.join(tr_dir, f"{name}_transcript.json")
    txt_path = os.path.join(txt_dir, f"{name}_text_emotions.json")

    if not os.path.exists(face_path):
        raise FileNotFoundError(f"Falta: {face_path}")
    if not os.path.exists(tr_path):
        raise FileNotFoundError(f"Falta: {tr_path}")

    face = read_json(face_path)
    tr = read_json(tr_path)

    text_emotions = None
    if os.path.exists(txt_path):
        text_emotions = read_json(txt_path)
    else:
        log.info(f"[{name}] No existe text_emotions, se fusiona solo con transcript.")

    # Paso 1: sync
    sync_data = sync_face_with_text_segments(face, tr, text_emotions)

    # Paso 2: build multimodal final
    multimodal = build_multimodal_from_sync(sync_data, video_name=name)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_multimodal.json")
    write_json(multimodal, out_path)

    return out_path


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Día 3 Paso 3: Pipeline end-to-end (sync + merge)")
    ap.add_argument("--names", nargs="*", default=None,
                    help="Lista de nombres base (ej: prueba1 prueba2). Si no se da, autodetecta por face_dir.")
    ap.add_argument("--face-dir", default="outputs/face_emotions", help="Carpeta face_emotions")
    ap.add_argument("--tr-dir", default="outputs/transcripts", help="Carpeta transcripts")
    ap.add_argument("--txt-dir", default="outputs/text_emotions", help="Carpeta text_emotions")
    ap.add_argument("--out-dir", default="outputs/multimodal", help="Salida multimodal")
    args = ap.parse_args()

    names = args.names
    if not names:
        # autodetecta a partir de los archivos en face-dir
        files = [f for f in os.listdir(args.face_dir) if f.endswith("_face_timeseries.json")]
        names = sorted([f.replace("_face_timeseries.json", "") for f in files])

    if not names:
        raise SystemExit("No se detectaron videos para procesar.")

    log.info(f"Procesando {len(names)} videos: {', '.join(names)}")

    ok = 0
    fail = 0

    for name in names:
        try:
            out_path = process_one(
                name=name,
                face_dir=args.face_dir,
                tr_dir=args.tr_dir,
                txt_dir=args.txt_dir,
                out_dir=args.out_dir
            )
            log.info(f"[{name}] ✅ OK -> {out_path}")
            ok += 1
        except Exception as e:
            log.info(f"[{name}] ❌ FAIL -> {e}")
            fail += 1

    log.info(f"Resumen: OK={ok} | FAIL={fail}")


if __name__ == "__main__":
    main()
