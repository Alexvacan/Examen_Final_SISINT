import os
from moviepy import VideoFileClip

# Rutas
video_path = "data/raw_videos/prueba4.mp4"
audio_out = "outputs/audio/prueba4.wav"

# Crear carpeta si no existe
os.makedirs(os.path.dirname(audio_out), exist_ok=True)

# Extraer audio
clip = VideoFileClip(video_path)
clip.audio.write_audiofile(
    audio_out,
    fps=16000,          # frecuencia ideal para Whisper
    nbytes=2,           # 16 bits
    codec="pcm_s16le"   # WAV sin compresión
)

print("✅ Audio extraído correctamente en:", audio_out)
