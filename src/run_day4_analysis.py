import os
from typing import Dict, Any

from video_utils import read_json, write_json
from logger_utils import get_logger

from day4_detect_changes import detect_changes
from day4_metrics import congruence_face_vs_text, congruence_vs_manual_labels
from day4_insights import generate_insights

log = get_logger("day4")


def analyze_one(video_name: str,
                multimodal_dir: str,
                labels_dir: str,
                out_dir: str) -> str:
    mm_path = os.path.join(multimodal_dir, f"{video_name}_multimodal.json")

    # Tus labels están como "Prueba1_labels.json" (con P mayúscula)
    # pero tus multimodal están como "prueba1_multimodal.json" (minúscula).
    # Esto lo hacemos robusto:
    candidates = [
        os.path.join(labels_dir, f"{video_name}_labels.json"),
        os.path.join(labels_dir, f"{video_name.capitalize()}_labels.json"),
        os.path.join(labels_dir, f"{video_name.upper()}_labels.json"),
    ]
    labels_path = None
    for c in candidates:
        if os.path.exists(c):
            labels_path = c
            break

    if not os.path.exists(mm_path):
        raise FileNotFoundError(f"Falta multimodal: {mm_path}")
    if labels_path is None:
        raise FileNotFoundError(f"No se encontró labels para {video_name} en {labels_dir}")

    multimodal = read_json(mm_path)

    changes = detect_changes(multimodal)
    m_face_text = congruence_face_vs_text(multimodal)
    m_manual = congruence_vs_manual_labels(multimodal, labels_path)

    insights = generate_insights(video_name, changes, m_face_text, m_manual)

    report: Dict[str, Any] = {
        "video": video_name,
        "changes": changes,
        "metrics": {
            "face_vs_text": m_face_text,
            "vs_manual_labels": m_manual
        },
        "insights": insights
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_name}_analysis.json")
    write_json(report, out_path)

    return out_path


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Día 4: análisis temporal + métricas + insights")
    ap.add_argument("--multimodal-dir", default="outputs/multimodal", help="carpeta outputs/multimodal")
    ap.add_argument("--labels-dir", default="data/labels", help="carpeta data/labels")
    ap.add_argument("--out-dir", default="outputs/day4", help="carpeta outputs/day4")
    ap.add_argument("--names", nargs="*", default=None, help="ej: prueba1 prueba2 ... (si no, autodetecta)")
    args = ap.parse_args()

    names = args.names
    if not names:
        files = [f for f in os.listdir(args.multimodal_dir) if f.endswith("_multimodal.json")]
        names = sorted([f.replace("_multimodal.json", "") for f in files])

    if not names:
        raise SystemExit("No se detectaron multimodal outputs para analizar.")

    ok = 0
    fail = 0

    for n in names:
        try:
            out_path = analyze_one(n, args.multimodal_dir, args.labels_dir, args.out_dir)
            log.info(f"[{n}] ✅ OK -> {out_path}")
            ok += 1
        except Exception as e:
            log.info(f"[{n}] ❌ FAIL -> {e}")
            fail += 1

    log.info(f"Resumen Día 4: OK={ok} | FAIL={fail}")


if __name__ == "__main__":
    main()
