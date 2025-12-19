import os
import json
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DAY4_DIR = os.path.join(BASE_DIR, "outputs", "day4")
REPORT_DIR = os.path.join(DAY4_DIR, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figs")


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def read_analysis_files():
    rows = []
    files = []
    for f in sorted(os.listdir(DAY4_DIR)):
        if not f.endswith("_analysis.json"):
            continue
        files.append(os.path.join(DAY4_DIR, f))
    return files


def build_summary(files):
    rows = []
    for path in files:
        name = os.path.basename(path).replace("_analysis.json", "")
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)

        metrics = d.get("metrics", {})
        face_vs_text = metrics.get("face_vs_text", {})
        face_manual = metrics.get("vs_manual_labels") or metrics.get("vs_manual_labels") or metrics.get("vs_manual_labels", {})
        # compatibility: some files may use 'vs_manual_labels' or 'metrics.face_vs_manual'
        face_vs_manual_metrics = metrics.get("face_vs_manual") or metrics.get("vs_manual_labels") or {}

        face_vs_text_val = face_vs_text.get("match_rate") if face_vs_text.get("match_rate") is not None else None
        face_vs_manual_val = face_vs_manual_metrics.get("match_rate") if face_vs_manual_metrics.get("match_rate") is not None else None

        rows.append({
            "video": name,
            "n_face_changes": d.get("changes", {}).get("n_face_changes"),
            "n_text_changes": d.get("changes", {}).get("n_text_changes"),
            "face_vs_text_match_rate": face_vs_text_val,
            "face_vs_text_total_with_text": face_vs_text.get("total_with_text"),
            "face_vs_manual_match_rate": face_vs_manual_val,
            "face_summary_num_changes": d.get("face_summary", {}).get("num_changes")
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(REPORT_DIR, "summary_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote summary: {out_csv}")
    return df


def plot_timeseries(path, name):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    raw = d.get("timeseries", {}).get("face_raw")
    smooth = d.get("timeseries", {}).get("face_smoothed")
    if not raw and not smooth:
        return False

    plt.figure(figsize=(10,3))
    if raw:
        plt.plot(raw, label="Raw Face", alpha=0.6)
    if smooth:
        plt.plot(smooth, label="Smoothed Face", linestyle='--')
    plt.legend()
    plt.title(f"Face time series - {name}")
    plt.xlabel("Frame")
    plt.ylabel("Emotion (code)")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"{name}_timeseries.png")
    plt.savefig(out)
    plt.close()
    return out


def plot_match_bar(df):
    names = df["video"].tolist()
    vals = [ (v if v is not None else 0.0) for v in df["face_vs_manual_match_rate"].tolist() ]
    # convert to percentages
    vals_pct = [v*100 for v in vals]
    plt.figure(figsize=(8,4))
    plt.bar(names, vals_pct, color='tab:blue')
    plt.ylabel("Match vs Manual (%)")
    plt.title("Face vs Manual match rate per video")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "face_vs_manual_bar.png")
    plt.savefig(out)
    plt.close()
    return out


def plot_changes_by_emotion(files):
    # For each video, count 'to' emotions from face_changes
    for path in files:
        name = os.path.basename(path).replace("_analysis.json", "")
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        changes = d.get("changes", {}).get("face_changes", [])
        if not changes:
            continue
        to_counts = Counter([c.get("to") for c in changes if c.get("to")])
        emotions = list(to_counts.keys())
        counts = [to_counts[e] for e in emotions]
        plt.figure(figsize=(6,3))
        plt.bar(emotions, counts, color='tab:orange')
        plt.title(f"Face changes by emotion - {name}")
        plt.ylabel("Count")
        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"{name}_changes_by_emotion.png")
        plt.savefig(out)
        plt.close()


def main():
    if not os.path.isdir(DAY4_DIR):
        raise SystemExit(f"No existe {DAY4_DIR}")
    ensure_dirs()
    files = read_analysis_files()
    if not files:
        raise SystemExit(f"No se encontraron archivos _analysis.json en {DAY4_DIR}")

    df = build_summary(files)

    # Per-video time series and changes plots
    saved = []
    for p in files:
        name = os.path.basename(p).replace("_analysis.json", "")
        ts_out = plot_timeseries(p, name)
        if ts_out:
            saved.append(ts_out)
    # Aggregate bar
    bar = plot_match_bar(df)
    saved.append(bar)
    # Changes by emotion
    plot_changes_by_emotion(files)

    print("Saved figures:")
    for s in sorted(os.listdir(FIG_DIR)):
        print(" -", os.path.join(FIG_DIR, s))


if __name__ == "__main__":
    main()
