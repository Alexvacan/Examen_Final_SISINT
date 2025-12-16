import os
import argparse

from face_emotion_day2 import analyze_frames_dir
from video_utils import list_subdirs, write_json
from logger_utils import get_logger


def main():
    log = get_logger("run_face_emotions_day2")

    ap = argparse.ArgumentParser(description="Día 2 (A): Emociones faciales por frames (serie temporal)")
    ap.add_argument(
        "--frames-root",
        default="data/extracted_frames",
        help="Carpeta raíz con subcarpetas por video (ej: data/extracted_frames)"
    )
    ap.add_argument(
        "--video-folder",
        default=None,
        help="Nombre de subcarpeta a procesar (ej: prueba1). Si no se da, procesa todas."
    )
    ap.add_argument(
        "--out-dir",
        default="outputs/face_emotions",
        help="Carpeta de salida (ej: outputs/face_emotions)"
    )
    ap.add_argument("--no-enhance", action="store_true", help="Desactiva CLAHE")
    ap.add_argument(
        "--enforce-detection",
        action="store_true",
        help="Si se activa, DeepFace fallará cuando no detecte rostro (NO recomendado)"
    )
    args = ap.parse_args()

    frames_root = args.frames_root
    out_dir = args.out_dir
    enhance = not args.no_enhance
    enforce_detection = args.enforce_detection

    if not os.path.isdir(frames_root):
        raise SystemExit(f"No existe frames-root: {frames_root}")

    if args.video_folder:
        targets = [args.video_folder]
    else:
        targets = list_subdirs(frames_root)

    if not targets:
        raise SystemExit(f"No hay subcarpetas para procesar en: {frames_root}")

    os.makedirs(out_dir, exist_ok=True)

    for folder in targets:
        frames_dir = os.path.join(frames_root, folder)
        if not os.path.isdir(frames_dir):
            log.info(f"Saltando (no existe carpeta): {frames_dir}")
            continue

        out_path = os.path.join(out_dir, f"{folder}_face_timeseries.json")

        log.info(f"Procesando: {frames_dir}")
        data = analyze_frames_dir(
            frames_dir=frames_dir,
            enhance=enhance,
            enforce_detection=enforce_detection
        )

        write_json(data, out_path)

        items = data.get("items", [])
        n_errors = sum(1 for x in items if isinstance(x, dict) and "error" in x)
        log.info(f"Guardado: {out_path}")
        log.info(f"Frames: {data.get('n_frames', 0)} | Registros: {len(items)} | Errores: {n_errors}")


if __name__ == "__main__":
    main()
