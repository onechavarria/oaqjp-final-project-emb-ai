"""Emotion detection using IBM Watson NLP."""

import requests


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion."""
    url = (
        "https://sn-watson-emotion.labs.skills.network/v1/"
        "watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    headers = {
        "grpc-metadata-mm-model-id":
        "emotion_aggregated-workflow_lang_en_stock"
    }

    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=20
    )

    emotions = ("anger", "disgust", "fear", "joy", "sadness")

    if response.status_code == 400:
        result = {emotion: None for emotion in emotions}
        result["dominant_emotion"] = None
        return result

    response.raise_for_status()

    data = response.json()
    scores = data["emotionPredictions"][0]["emotion"]

    result = {emotion: scores[emotion] for emotion in emotions}
    result["dominant_emotion"] = max(
        emotions,
        key=lambda emotion: result[emotion]
    )

    return result
