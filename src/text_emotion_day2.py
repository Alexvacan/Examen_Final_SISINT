from transformers import pipeline
from typing import Dict, Any
from video_utils import read_json, write_json
import argparse
import os

# 1) Cargar modelo UNA SOLA VEZ (por defecto, se puede sobreescribir via --model)
emotion_pipe = None


def get_emotion_pipe(model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
    global emotion_pipe
    if emotion_pipe is None:
        emotion_pipe = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True
        )
    return emotion_pipe


# 2) Función para extraer emoción dominante
def get_text_emotion(text: str) -> Dict[str, Any]:
    pipe = get_emotion_pipe()
    res = pipe(text)[0]
    best = max(res, key=lambda x: x["score"])
    return {
        "emotion": best["label"],
        "confidence": float(best["score"])
    }


# 3) Procesar archivo multimodal
def enrich_multimodal_with_text_emotion(multimodal_path: str, out_path: str):
    data = read_json(multimodal_path)

    for it in data.get("items", []):
        txt = it.get("text")

        if isinstance(txt, dict):
            raw = txt.get("raw") or txt.get("text")
        else:
            raw = txt

        if not raw or not str(raw).strip():
            continue

        try:
            emo = get_text_emotion(raw)
            it["text_emotion"] = emo
        except Exception as e:
            it["text_emotion"] = {"error": str(e)}

    write_json(data, out_path)


def analyze_text_emotions(transcript_json: Dict[str, Any], model_name: str = None) -> Dict[str, Any]:
    # Mantener compatibilidad con run_text_emotions_day2.py
    if model_name:
        get_emotion_pipe(model_name)

    data = {"items": [], "n_segments": 0}
    items = transcript_json.get("items", [])
    for it in items:
        text = None
        if isinstance(it, dict):
            text = it.get("text") or it.get("raw")
        else:
            text = it

        if not text or not str(text).strip():
            data["items"].append({"error": "empty_text"})
            continue

        try:
            emo = get_text_emotion(text)
            out = {**(it if isinstance(it, dict) else {}), "text_emotion": emo}
            data["items"].append(out)
        except Exception as e:
            data["items"].append({"error": str(e)})

    data["n_segments"] = len(items)
    return data


def main():
    import logging
    from logger_utils import get_logger

    log = get_logger("text_emotion_day2")

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", help="JSON multimodal input path")
    ap.add_argument("--out", dest="out_path", help="JSON output path")
    ap.add_argument("--model", dest="model", default="j-hartmann/emotion-english-distilroberta-base")
    args = ap.parse_args()

    if args.in_path and args.out_path:
        get_emotion_pipe(args.model)
        log.info(f"Procesando {args.in_path} -> {args.out_path}")
        enrich_multimodal_with_text_emotion(args.in_path, args.out_path)
        log.info("Listo")
    else:
        print("Uso: python src/text_emotion_day2.py --in <multimodal.json> --out <out.json>")


if __name__ == "__main__":
    main()
from typing import Dict, Any, List
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def analyze_text_emotions(
    transcript_json: Dict[str, Any],
    model_name: str = "pysentimiento/robertuito-emotion-analysis"
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,          # devuelve todas las clases
        truncation=True
    )

    segments = transcript_json.get("segments", [])
    results: List[Dict[str, Any]] = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        try:
            scores = classifier(text)[0]  # lista de {label, score}
            scores_dict = {x["label"]: float(x["score"]) for x in scores}
            dominant = max(scores_dict, key=scores_dict.get)

            results.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": text,
                "dominant_emotion": dominant,
                "scores": scores_dict
            })
        except Exception as e:
            results.append({
                "start": float(seg.get("start", -1)),
                "end": float(seg.get("end", -1)),
                "text": text,
                "error": str(e)
            })

    return {"n_segments": len(results), "items": results}
