import json
import matplotlib.pyplot as plt
import os
import argparse

def load_series(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    raw = d.get("timeseries", {}).get("face_raw", [])
    smooth = d.get("timeseries", {}).get("face_smoothed", [])
    return raw, smooth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to a single *_analysis.json in outputs/day4")
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        raise SystemExit(f"No existe archivo: {args.path}")

    raw, smooth = load_series(args.path)

    plt.figure(figsize=(10,4))
    plt.plot(raw, label="Raw Face", linestyle='-', alpha=0.7)
    plt.plot(smooth, label="Smoothed Face", linestyle='--', linewidth=2)
    plt.legend()
    plt.title("Evolución temporal de emociones faciales")
    plt.xlabel("Frame")
    plt.ylabel("Emoción (codificada)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
