import os
import json
import argparse
from pathlib import Path
from faster_whisper import WhisperModel


def transcribe_one(model: WhisperModel, audio_path: str, out_json: str, language: str = "es") -> str:
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True
    )

    data = {
        "audio": audio_path,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segments": []
    }

    for s in segments:
        data["segments"].append({
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip()
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_json


def main():
    ap = argparse.ArgumentParser(description="Transcribir WAV usando faster-whisper")
    ap.add_argument("--audio-dir", default="outputs/audio", help="Carpeta con audios .wav")
    ap.add_argument("--out-dir", default="outputs/transcripts", help="Salida JSON transcripts")
    ap.add_argument("--model", default="small", help="Modelo whisper (tiny/base/small/medium/large-v3)")
    ap.add_argument("--device", default="cpu", help="cpu o cuda")
    ap.add_argument("--compute-type", default="int8", help="int8 / float16 / float32")
    ap.add_argument("--language", default="es", help="Idioma (es)")
    args = ap.parse_args()

    audio_dir = args.audio_dir
    out_dir = args.out_dir

    audios = sorted([str(p) for p in Path(audio_dir).glob("*.wav")])
    if not audios:
        raise SystemExit(f"No se encontraron .wav en: {audio_dir}")

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    for apath in audios:
        name = Path(apath).stem.lower()
        out_json = os.path.join(out_dir, f"{name}_transcript.json")
        transcribe_one(model, apath, out_json, language=args.language)
        print(f"✅ Transcript: {name} -> {out_json}")


if __name__ == "__main__":
    main()
