import os
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Día 2 (D): Crear template de etiquetas manuales"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Ruta del video (ej: data/raw_videos/Prueba1.mp4)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Ruta de salida del JSON de etiquetas (opcional)"
    )
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        raise SystemExit(f"❌ No existe el video: {video_path}")

    video_name = Path(video_path).stem
    out_path = args.out or f"data/labels/{video_name}_labels.json"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    template = {
        "video": video_path,
        "fps_note": "timestamps en segundos",
        "segments": [
            {
                "start": 0.0,
                "end": 0.0,
                "emotion": "neutral",
                "note": "REEMPLAZAR con rangos reales"
            }
        ]
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print("✅ Archivo de etiquetas creado:")
    print(out_path)
    print("✏️  Ahora edita el archivo y ajusta los tiempos y emociones.")


if __name__ == "__main__":
    main()
