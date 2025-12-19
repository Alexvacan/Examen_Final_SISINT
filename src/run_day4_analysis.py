import os
from typing import Dict, Any, Optional

from video_utils import read_json, write_json
from logger_utils import get_logger

from day4_detect_changes import detect_changes
from day4_metrics import congruence_face_vs_text, congruence_vs_manual_labels
from day4_insights import generate_insights

log = get_logger("day4")


def _find_case_insensitive(path_dir: str, filename: str) -> Optional[str]:
    """
    Busca filename dentro de path_dir ignorando mayúsculas/minúsculas.
    Retorna ruta real si existe.
    """
    target = filename.lower()
    try:
        for f in os.listdir(path_dir):
            if f.lower() == target:
                return os.path.join(path_dir, f)
    except FileNotFoundError:
        return None
    return None


def analyze_one(video_name: str,
                multimodal_dir: str,
                labels_dir: str,
                out_dir: str) -> str:
    mm_filename = f"{video_name}_multimodal.json"
    mm_path = _find_case_insensitive(multimodal_dir, mm_filename)

    if mm_path is None:
        raise FileNotFoundError(f"Falta multimodal: {os.path.join(multimodal_dir, mm_filename)}")

    # labels: busca robusto por cualquier case
    labels_filename = f"{video_name}_labels.json"
    labels_path = _find_case_insensitive(labels_dir, labels_filename)

    if labels_path is None:
        raise FileNotFoundError(f"No se encontró labels para {video_name} en {labels_dir} (esperaba algo como {labels_filename})")

    multimodal = read_json(mm_path)

    changes = detect_changes(multimodal)
    m_face_text = congruence_face_vs_text(multimodal)
    m_manual = congruence_vs_manual_labels(multimodal, labels_path)

    insights = generate_insights(video_name, changes, m_face_text, m_manual)

    report: Dict[str, Any] = {
        "video": video_name,
        "multimodal_path": mm_path,
        "labels_path": labels_path,
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
        if not os.path.exists(args.multimodal_dir):
            raise SystemExit(f"No existe la carpeta multimodal: {args.multimodal_dir}")

        files = [f for f in os.listdir(args.multimodal_dir) if f.lower().endswith("_multimodal.json")]
        names = sorted([f[:-len("_multimodal.json")] for f in files])

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
            log.error(f"[{n}] ❌ FAIL -> {e}", exc_info=True)
            fail += 1

    log.info(f"Resumen Día 4: OK={ok} | FAIL={fail}")


if __name__ == "__main__":
    main()
