import os
import argparse
from pathlib import Path
import cv2


def extract_frames(video_path: str, out_dir: str, target_fps: float = 2.0) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0  # fallback

    step = max(1, int(round(fps / target_fps)))

    i = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            t = i / fps
            name = f"frame_{i:06d}_t{t:.2f}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), frame)
            saved += 1
        i += 1

    cap.release()
    return {"video": video_path, "fps": fps, "saved": saved, "out_dir": out_dir, "step": step}


def main():
    ap = argparse.ArgumentParser(description="Extraer frames de videos")
    ap.add_argument("--videos-dir", default="data/raw_videos", help="Carpeta con videos .mp4")
    ap.add_argument("--out-root", default="data/extracted_frames", help="Carpeta raíz de salida")
    ap.add_argument("--names", nargs="*", default=None, help="Nombres base a procesar (sin .mp4). Si no, procesa todos.")
    ap.add_argument("--target-fps", type=float, default=2.0, help="Frames por segundo a guardar (ej 2.0)")
    args = ap.parse_args()

    vdir = args.videos_dir
    out_root = args.out_root

    if args.names:
        videos = [os.path.join(vdir, f"{n}.mp4") for n in args.names]
    else:
        videos = sorted([str(p) for p in Path(vdir).glob("*.mp4")])

    if not videos:
        raise SystemExit(f"No se encontraron videos .mp4 en: {vdir}")

    for vp in videos:
        name = Path(vp).stem
        out_dir = os.path.join(out_root, name.lower())
        info = extract_frames(vp, out_dir, target_fps=args.target_fps)
        print(f"✅ Frames: {name} | fps={info['fps']:.2f} | saved={info['saved']} | out={out_dir}")


if __name__ == "__main__":
    main()
