import os
import json
from typing import Dict, Any, List

from transformers import pipeline


def load_transcript(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_text_emotions(
    transcript_json: Dict[str, Any],
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
) -> Dict[str, Any]:
    """
    Analiza emoción del texto por segmento usando Transformer preentrenado.
    Devuelve estructura lista para integración posterior.
    """
    classifier = pipeline(
        "text-classification",
        model=model_name,
        return_all_scores=True
    )

    segments = transcript_json.get("segments", [])
    results: List[Dict[str, Any]] = []

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        try:
            scores = classifier(text)[0]
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

    return {
        "n_segments": len(results),
        "items": results
    }


def save_json(data: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
