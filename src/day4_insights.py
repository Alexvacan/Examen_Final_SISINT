from typing import Dict, Any, List, Optional


def generate_insights(video_name: str,
                      changes: Dict[str, Any],
                      m_face_text: Dict[str, Any],
                      m_manual: Dict[str, Any]) -> List[str]:
    insights: List[str] = []

    # Cambios frecuentes
    n_face = changes.get("n_face_changes", 0)
    n_text = changes.get("n_text_changes", 0)

    if n_face >= 6:
        insights.append(f"[{video_name}] Alta variación facial: {n_face} cambios (posible inestabilidad emocional).")
    elif n_face >= 3:
        insights.append(f"[{video_name}] Variación facial moderada: {n_face} cambios.")
    else:
        insights.append(f"[{video_name}] Variación facial baja: {n_face} cambios.")

    if n_text >= 3:
        insights.append(f"[{video_name}] Variación emocional en texto detectada: {n_text} cambios.")

    # Congruencia cara vs texto
    rate_ft = m_face_text.get("match_rate", None)
    if rate_ft is None:
        insights.append(f"[{video_name}] No hay suficientes puntos con emoción de texto para medir congruencia cara-texto.")
    else:
        if rate_ft >= 0.65:
            insights.append(f"[{video_name}] Alta congruencia cara-texto (match_rate={rate_ft:.2f}).")
        elif rate_ft >= 0.40:
            insights.append(f"[{video_name}] Congruencia cara-texto media (match_rate={rate_ft:.2f}).")
        else:
            insights.append(f"[{video_name}] Baja congruencia cara-texto (match_rate={rate_ft:.2f}).")

    # Validación contra etiquetas manuales
    rate_m = m_manual.get("match_rate", None)
    if rate_m is None:
        insights.append(f"[{video_name}] No hay suficientes segmentos etiquetados para evaluar contra ground-truth.")
    else:
        if rate_m >= 0.70:
            insights.append(f"[{video_name}] Buen acuerdo con etiquetas manuales (match_rate={rate_m:.2f}).")
        elif rate_m >= 0.50:
            insights.append(f"[{video_name}] Acuerdo medio con etiquetas manuales (match_rate={rate_m:.2f}).")
        else:
            insights.append(f"[{video_name}] Acuerdo bajo con etiquetas manuales (match_rate={rate_m:.2f}).")

    return insights
