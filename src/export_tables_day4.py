import json
import os
import pandas as pd

DAY4_DIR = os.path.join("outputs", "day4")

def main():
    rows = []

    if not os.path.isdir(DAY4_DIR):
        raise SystemExit(f"No existe {DAY4_DIR}")

    for f in sorted(os.listdir(DAY4_DIR)):
        if not f.endswith("_analysis.json"):
            continue

        path = os.path.join(DAY4_DIR, f)
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)

        face_text = d.get("metrics", {}).get("face_vs_text", {})
        face_manual = d.get("metrics", {}).get("face_vs_manual", {})

        face_vs_text_val = (
            face_text.get("match_rate")
            if face_text.get("match_rate") is not None
            else "N/A"
        )

        face_vs_manual_val = face_manual.get("match_rate")
        face_vs_manual_str = (
            f"{face_vs_manual_val*100:.2f}%" if face_vs_manual_val is not None else "N/A"
        )

        rows.append({
            "video": f.replace("_analysis.json", ""),
            "face_changes": d.get("face_summary", {}).get("num_changes"),
            "face_vs_text": face_vs_text_val,
            "face_vs_manual": face_vs_manual_str
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(DAY4_DIR, "summary_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
