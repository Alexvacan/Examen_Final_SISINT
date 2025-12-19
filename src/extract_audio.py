import os
import argparse
from pathlib import Path
from moviepy import VideoFileClip


def extract_audio(video_path: str, audio_out: str) -> str:
    os.makedirs(os.path.dirname(audio_out), exist_ok=True)
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise RuntimeError(f"El video no tiene audio: {video_path}")

    clip.audio.write_audiofile(
    audio_out,
    fps=16000,
    nbytes=2,
    codec="pcm_s16le"
    )

    clip.close()
    return audio_out


def main():
    ap = argparse.ArgumentParser(description="Extraer audio WAV (16kHz) desde videos")
    ap.add_argument("--videos-dir", default="data/raw_videos", help="Carpeta con videos .mp4")
    ap.add_argument("--out-dir", default="outputs/audio", help="Carpeta salida de audios")
    ap.add_argument("--names", nargs="*", default=None, help="Nombres base a procesar (sin .mp4). Si no, procesa todos.")
    args = ap.parse_args()

    vdir = args.videos_dir
    out_dir = args.out_dir

    if args.names:
        videos = [os.path.join(vdir, f"{n}.mp4") for n in args.names]
    else:
        videos = sorted([str(p) for p in Path(vdir).glob("*.mp4")])

    if not videos:
        raise SystemExit(f"No se encontraron videos .mp4 en: {vdir}")

    for vp in videos:
        name = Path(vp).stem.lower()
        audio_out = os.path.join(out_dir, f"{name}.wav")
        extract_audio(vp, audio_out)
        print(f"✅ Audio: {name} -> {audio_out}")


if __name__ == "__main__":
    main()
