import os
import argparse

from video_utils import list_subdirs, read_json, write_json
from logger_utils import get_logger
from text_emotion_day2 import analyze_text_emotions


def main():
    log = get_logger("run_text_emotions_day2")

    ap = argparse.ArgumentParser(description="Día 2 (B): Emoción de texto con Transformers (por segmento)")
    ap.add_argument(
        "--transcripts-dir",
        default="outputs/transcripts",
        help="Carpeta con JSON de transcripciones (ej: outputs/transcripts)"
    )
    ap.add_argument(
        "--out-dir",
        default="outputs/text_emotions",
        help="Carpeta de salida (ej: outputs/text_emotions)"
    )
    ap.add_argument(
        "--model",
        default="j-hartmann/emotion-english-distilroberta-base",
        help="Modelo HF para emotion (puedes cambiarlo)"
    )
    args = ap.parse_args()

    transcripts_dir = args.transcripts_dir
    out_dir = args.out_dir
    model_name = args.model

    if not os.path.isdir(transcripts_dir):
        raise SystemExit(f"No existe transcripts-dir: {transcripts_dir}")

    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(transcripts_dir) if f.endswith("_transcript.json")])
    if not files:
        raise SystemExit(f"No se encontraron *_transcript.json en: {transcripts_dir}")

    for fname in files:
        base = fname.replace("_transcript.json", "")
        in_path = os.path.join(transcripts_dir, fname)
        out_path = os.path.join(out_dir, f"{base}_text_emotions.json")

        log.info(f"Procesando: {in_path}")
        transcript = read_json(in_path)

        data = analyze_text_emotions(
            transcript_json=transcript,
            model_name=model_name
        )

        write_json(data, out_path)

        items = data.get("items", [])
        n_errors = sum(1 for x in items if isinstance(x, dict) and "error" in x)
        log.info(f"Guardado: {out_path}")
        log.info(f"Segmentos: {data.get('n_segments', 0)} | Errores: {n_errors}")


if __name__ == "__main__":
    main()
