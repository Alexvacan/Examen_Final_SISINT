import os
import argparse

from face_emotion_day2 import analyze_frames_dir, save_json


def main():
    ap = argparse.ArgumentParser(description="Día 2 (A): Emociones faciales por frames (serie temporal)")
    ap.add_argument("--frames-root", default="data/extracted_frames",
                    help="Carpeta raíz que contiene subcarpetas por video (ej: data/extracted_frames)")
    ap.add_argument("--video-folder", default=None,
                    help="Nombre de subcarpeta a procesar (ej: prueba1). Si no se da, procesa todas.")
    ap.add_argument("--out-dir", default="outputs/face_emotions",
                    help="Carpeta de salida (ej: outputs/face_emotions)")
    ap.add_argument("--no-enhance", action="store_true",
                    help="Desactiva CLAHE")
    ap.add_argument("--enforce-detection", action="store_true",
                    help="Si se activa, DeepFace fallará cuando no detecte rostro (NO recomendado)")
    args = ap.parse_args()

    frames_root = args.frames_root
    out_dir = args.out_dir
    enhance = not args.no_enhance
    enforce_detection = args.enforce_detection

    if args.video_folder:
        targets = [args.video_folder]
    else:
        # procesa todas las subcarpetas en frames_root
        targets = [d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))]
        targets.sort()

    if not targets:
        raise SystemExit(f"No hay subcarpetas para procesar en: {frames_root}")

    for folder in targets:
        frames_dir = os.path.join(frames_root, folder)
        out_path = os.path.join(out_dir, f"{folder}_face_timeseries.json")

        print(f"\n== Procesando: {frames_dir}")
        data = analyze_frames_dir(
            frames_dir=frames_dir,
            enhance=enhance,
            enforce_detection=enforce_detection
        )
        save_json(data, out_path)

        n_items = len(data.get("items", []))
        n_errors = sum(1 for x in data.get("items", []) if "error" in x)
        print(f"✅ Guardado: {out_path}")
        print(f"   Frames: {data.get('n_frames')} | Registros: {n_items} | Errores: {n_errors}")


if __name__ == "__main__":
    main()
