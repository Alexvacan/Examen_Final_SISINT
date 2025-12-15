import os
import json
import cv2
from deepface import DeepFace

# -----------------------------
# Configuración
# -----------------------------
frames_dir = "data/extracted_frames/prueba1"
out_json = "outputs/face_emotions/prueba1_face_emotions.json"
os.makedirs(os.path.dirname(out_json), exist_ok=True)

# -----------------------------
# Función para mejorar iluminación
# -----------------------------
def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

# -----------------------------
# Utilidad para JSON
# -----------------------------
def to_float_dict(d):
    clean = {}
    for k, v in d.items():
        try:
            clean[k] = float(v)
        except Exception:
            clean[k] = v
    return clean

# -----------------------------
# Selección de frames
# - descarta primeros 3 segundos
# - usa hasta 60 frames
# -----------------------------
all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

frames = []
for f in all_frames:
    try:
        t = float(f.split("_t")[1].replace(".jpg", ""))
        if t >= 3.0:
            frames.append(f)
    except:
        pass

frames = frames[:60]

# -----------------------------
# Análisis de emociones
# -----------------------------
results = []

for f in frames:
    path = os.path.join(frames_dir, f)

    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Imagen no pudo cargarse")

        img = enhance(img)

        r = DeepFace.analyze(
            img_path=img,
            actions=["emotion"],
            enforce_detection=False
        )

        if isinstance(r, list):
            r = r[0]

        results.append({
            "frame": f,
            "dominant_emotion": r.get("dominant_emotion"),
            "scores": to_float_dict(r.get("emotion", {}))
        })

    except Exception as e:
        results.append({
            "frame": f,
            "error": str(e)
        })

# -----------------------------
# Guardar resultados
# -----------------------------
with open(out_json, "w", encoding="utf-8") as fp:
    json.dump(results, fp, indent=2, ensure_ascii=False)

print(f"✅ Emociones guardadas en: {out_json}")
print("✅ Ejemplo (primer registro):", results[0] if results else "Sin datos")
