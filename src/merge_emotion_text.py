import json, os, re

emo_path = "outputs/face_emotions/prueba1_face_emotions.json"
txt_path = "outputs/transcripts/prueba1_transcript.json"
out_path = "outputs/prueba1_multimodal.json"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(emo_path, "r", encoding="utf-8") as f:
    emotions = json.load(f)

with open(txt_path, "r", encoding="utf-8") as f:
    transcript = json.load(f)

def frame_time(name):
    m = re.search(r"_t([0-9.]+)\.jpg$", name)
    return float(m.group(1)) if m else None

emotion_rows = []
for e in emotions:
    if "scores" not in e:
        continue
    t = frame_time(e["frame"])
    if t is None:
        continue
    emotion_rows.append({
        "t": t,
        "dominant_emotion": e["dominant_emotion"],
        "scores": e["scores"]
    })

segments = transcript.get("segments", [])

merged = []
for e in emotion_rows:
    t = e["t"]
    text = None
    for s in segments:
        if s["start"] <= t <= s["end"]:
            text = s["text"]
            break
    merged.append({
        "t": t,
        "emotion": e["dominant_emotion"],
        "text": text,
        "scores": e["scores"]
    })

with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"items": merged}, f, indent=2, ensure_ascii=False)

print("✅ Multimodal generado en:", out_path)
print("✅ Ejemplo:", merged[0] if merged else "VACÍO")
