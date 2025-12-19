import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from video_utils import read_json, normalize_ts


# Emociones típicas de DeepFace
FACE_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_multimodal(path: str) -> Dict[str, Any]:
    return read_json(path)


def load_labels(path: str) -> Dict[str, Any]:
    return read_json(path)


def label_at_t(labels: Dict[str, Any], t: float) -> Optional[str]:
    segs = labels.get("segments", [])
    for s in segs:
        st = normalize_ts(s.get("start"))
        en = normalize_ts(s.get("end"))
        emo = s.get("emotion")
        if st is None or en is None or not emo:
            continue
        if st <= t <= en:
            return str(emo).strip()
    return None


def extract_face_series(multimodal: Dict[str, Any]) -> List[Tuple[float, np.ndarray]]:
    items = multimodal.get("items", [])
    series: List[Tuple[float, np.ndarray]] = []

    for it in items:
        t = normalize_ts(it.get("t"))
        if t is None:
            continue
        face = it.get("face") or {}
        scores = (face.get("scores") or {})
        # vector en orden fijo
        vec = np.array([float(scores.get(k, 0.0)) for k in FACE_KEYS], dtype=np.float32)
        # normaliza a sum=1 (por si vienen como porcentajes)
        s = float(vec.sum())
        if s > 0:
            vec = vec / s
        series.append((t, vec))

    series.sort(key=lambda x: x[0])
    return series


def build_windows(series: List[Tuple[float, np.ndarray]],
                  labels: Dict[str, Any],
                  window: int = 10,
                  stride: int = 1) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    X: (N, window, 7)
    y: (N,) etiquetas string
    t_end: lista del t del último frame de cada ventana
    """
    X = []
    y = []
    t_end = []

    if len(series) < window:
        return np.zeros((0, window, len(FACE_KEYS)), dtype=np.float32), np.zeros((0,), dtype=object), []

    for i in range(0, len(series) - window + 1, stride):
        w = series[i:i+window]
        t_last = w[-1][0]
        gt = label_at_t(labels, t_last)
        if gt is None:
            continue  # sin etiqueta para entrenar

        X.append(np.stack([v for (_, v) in w], axis=0))
        y.append(gt)
        t_end.append(t_last)

    return np.array(X, dtype=np.float32), np.array(y, dtype=object), t_end


def build_model(input_shape: Tuple[int, int], n_classes: int) -> tf.keras.Model:
    """
    input_shape: (window, 7)
    """
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.GRU(32, return_sequences=False)(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    ap = argparse.ArgumentParser(description="Día 5: GRU temporal sobre scores faciales + labels manuales")
    ap.add_argument("--multimodal-dir", default="outputs/multimodal", help="Carpeta multimodal")
    ap.add_argument("--labels-dir", default="data/labels", help="Carpeta labels manuales")
    ap.add_argument("--names", nargs="*", default=None, help="Nombres base (sin _multimodal). Si no, autodetecta.")
    ap.add_argument("--window", type=int, default=10, help="Tamaño de ventana (frames)")
    ap.add_argument("--stride", type=int, default=1, help="Stride de ventanas")
    ap.add_argument("--epochs", type=int, default=2, help="Épocas (corto, para examen)")
    ap.add_argument("--batch", type=int, default=32, help="Batch size")
    ap.add_argument("--out-dir", default="outputs/day5_rnn", help="Salida reportes")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # autodetecta
    if not args.names:
        files = [f for f in os.listdir(args.multimodal_dir) if f.endswith("_multimodal.json")]
        args.names = sorted([f.replace("_multimodal.json", "") for f in files])

    if not args.names:
        raise SystemExit("No se detectaron multimodal outputs.")

    # juntamos data de todos los videos (mejor para entrenar algo decente)
    all_X = []
    all_y = []
    per_video_stats = {}

    for name in args.names:
        mm_path = os.path.join(args.multimodal_dir, f"{name}_multimodal.json")

        # labels robusto (igual que tu day4)
        candidates = [
            os.path.join(args.labels_dir, f"{name}_labels.json"),
            os.path.join(args.labels_dir, f"{name.capitalize()}_labels.json"),
            os.path.join(args.labels_dir, f"{name.upper()}_labels.json"),
        ]
        labels_path = next((c for c in candidates if os.path.exists(c)), None)

        if not os.path.exists(mm_path) or labels_path is None:
            per_video_stats[name] = {"used": False, "reason": "Falta multimodal o labels"}
            continue

        mm = load_multimodal(mm_path)
        labels = load_labels(labels_path)

        series = extract_face_series(mm)
        X, y, _t_end = build_windows(series, labels, window=args.window, stride=args.stride)

        per_video_stats[name] = {
            "used": True,
            "mm_path": mm_path,
            "labels_path": labels_path,
            "n_frames": len(series),
            "n_samples": int(X.shape[0]),
            "label_set": sorted(list(set(y.tolist()))) if X.shape[0] > 0 else []
        }

        if X.shape[0] > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        out = {
            "ok": False,
            "message": "No hay samples etiquetados suficientes. Revisa que tus labels tengan segmentos reales y que cubran el tiempo del video.",
            "per_video": per_video_stats
        }
        out_path = os.path.join(args.out_dir, "day5_gru_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("❌", out["message"])
        print("Reporte:", out_path)
        return

    X = np.concatenate(all_X, axis=0)
    y_str = np.concatenate(all_y, axis=0)

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    model = build_model((args.window, len(FACE_KEYS)), n_classes=len(le.classes_))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=1
    )

    # Eval final
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    # Guardar modelo y reporte
    model_path = os.path.join(args.out_dir, "gru_model.keras")
    model.save(model_path)

    report = {
        "ok": True,
        "window": args.window,
        "stride": args.stride,
        "epochs": args.epochs,
        "batch": args.batch,
        "face_keys": FACE_KEYS,
        "classes": le.classes_.tolist(),
        "n_samples_total": int(X.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "history": {
            "loss": [float(x) for x in history.history.get("loss", [])],
            "accuracy": [float(x) for x in history.history.get("accuracy", [])],
            "val_loss": [float(x) for x in history.history.get("val_loss", [])],
            "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
        },
        "per_video": per_video_stats
    }

    report_path = os.path.join(args.out_dir, "day5_gru_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ GRU entrenada. val_acc={val_acc:.3f} | samples={X.shape[0]}")
    print("✅ Modelo:", model_path)
    print("✅ Reporte:", report_path)


if __name__ == "__main__":
    main()
