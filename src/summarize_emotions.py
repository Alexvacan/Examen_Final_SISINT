import json
from collections import Counter

path = "outputs/face_emotions/prueba1_face_emotions.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# solo frames sin error
valid = [x for x in data if "scores" in x and x.get("dominant_emotion")]

dom = Counter([x["dominant_emotion"] for x in valid])

# promedio de scores
keys = list(valid[0]["scores"].keys())
avg = {k: 0.0 for k in keys}

for x in valid:
    for k in keys:
        avg[k] += float(x["scores"].get(k, 0.0))

n = len(valid) if valid else 1
avg = {k: v/n for k, v in avg.items()}

print("Frames válidos:", len(valid))
print("Dominante más frecuente:", dom.most_common(3))
print("Promedios (top 3):", sorted(avg.items(), key=lambda kv: kv[1], reverse=True)[:3])
