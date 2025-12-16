import os
import argparse

from text_emotion_day2 import load_transcript, analyze_text_emotions, save_json


def main():
    ap = argparse.ArgumentParser(description="Día 2 (B): Emoción de texto con Transformers")
    ap.add_argument("--transcripts-dir", default="outputs/transcripts",
                    help="Carpeta con JSON de transcripciones")
    ap.add_argument("--out-dir", default="outputs/text_emotions",
                    help="Carpeta de salida")
    args = ap.parse_args()

    transcripts_dir = args.transcripts_dir
    out_dir = args.out_dir

    files = sorted([
        f for f in os.listdir(transcripts_dir)
        if f.endswith("_transcript.json")
    ])

    if not files:
        raise SystemExit(f"No se encontraron transcripts en {transcripts_dir}")

    for f in files:
        name = f.replace("_transcript.json", "")
        in_path = os.path.join(transcripts_dir, f)
        out_path = os.path.join(out_dir, f"{name}_text_emotions.json")

        print(f"\n== Procesando texto: {in_path}")
        transcript = load_transcript(in_path)
        data = analyze_text_emotions(transcript)
        save_json(data, out_path)

        n = data.get("n_segments", 0)
        n_err = sum(1 for x in data.get("items", []) if "error" in x)
        print(f"✅ Guardado: {out_path}")
        print(f"   Segmentos: {n} | Errores: {n_err}")


if __name__ == "__main__":
    main()
