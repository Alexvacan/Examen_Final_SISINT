import os
import json
from faster_whisper import WhisperModel

audio_path = "outputs/audio/prueba1.wav"
out_json = "outputs/transcripts/prueba1_transcript.json"

os.makedirs(os.path.dirname(out_json), exist_ok=True)

# Modelo pequeño = buen balance calidad / velocidad
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(
    audio_path,
    language="es",
    vad_filter=True
)

data = {
    "language": info.language,
    "language_probability": info.language_probability,
    "segments": []
}

for s in segments:
    data["segments"].append({
        "start": float(s.start),
        "end": float(s.end),
        "text": s.text.strip()
    })

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Transcripción guardada en:", out_json)
print("✅ Primer segmento:", data["segments"][0] if data["segments"] else "VACÍO")
