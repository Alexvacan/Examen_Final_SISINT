# Interview Analyzer – Sistemas Inteligentes

Este proyecto implementa un sistema de análisis multimodal aplicado a entrevistas en video, combinando visión por computadora y procesamiento de lenguaje natural. El sistema permite extraer emociones faciales a partir de frames de video y sincronizarlas con el contenido textual obtenido mediante transcripción automática del audio.

## Estructura del proyecto

- data/
  - raw_videos/: videos originales
  - extracted_frames/: frames extraídos por video
  - extracted_audio/: audios extraídos en formato WAV
- outputs/
  - audio/: archivos de audio procesados
  - face_emotions/: resultados del análisis emocional facial
  - transcripts/: transcripciones automáticas
  - prueba1_multimodal.json: integración emoción-texto
- src/
  - extract_frames.py
  - extract_audio.py
  - face_emotion.py
  - summarize_emotions.py
  - transcribe.py
  - merge_emotion_text.py
  - video_utils.py
  - schemas.py
- notebooks/: pruebas exploratorias en Colab/Jupyter
- BITACORA_DIA1.md
- README.md

## Tecnologías utilizadas
- Python 3.11
- OpenCV
- DeepFace
- TensorFlow
- MoviePy
- Faster-Whisper
- HuggingFace

## Estado del proyecto
Día 1 completado: configuración del entorno, extracción de datos y generación de resultados multimodales iniciales.
