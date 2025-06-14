import logging
import os
from typing import List

import fasttext
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%d-%m %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Download fasttext language detection model if not downloaded
ft_model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
if not os.path.exists("lid.176.bin"):
    print(f"Downloading model from {ft_model_url}...")
    response = requests.get(ft_model_url, stream=True)
    response.raise_for_status()

    with open("lid.176.bin", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download completed.")
else:
    print("Model already exists.")


detector = fasttext.load_model("lid.176.bin")
embedder = SentenceTransformer("sentence-transformers/LaBSE")

code2lang = {
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "el": "Greek",
    "en": "English",
    "fa": "Persian",
    "ha": "Hausa",
    "ko": "Korean",
    "su": "Sundanese",
}


def detect_lang_batch(texts: List[str]) -> List[str]:
    """Helper function to detect text language in a batch. Uses Fasttext models."""
    langs = []
    for t in texts:
        result = detector.predict(t.replace("\n", " "))
        lang_code = result[0][0].replace("__label__", "")
        langs.append(code2lang.get(lang_code, lang_code))
    return langs


def semantic_similarity(pred: str, targets: list[str]) -> float:
    """Computes cosine similarity between pred and targets. Uses Sentence transformers models."""
    embeddings = embedder.encode(
        [pred] + targets.tolist(),
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    scores = util.cos_sim(embeddings[0], embeddings[1:]).flatten()
    try:
        return scores.max().item()
    except Exception as e:
        print(e, pred, targets, len(embeddings), scores)


def batch_lass(
    preds: List[str],
    target_lists: List[List[str]],
    target_langs: List[str],
    beta: float = 0.8,
) -> List[float]:
    """Calculates Language Aware Semantic Score (LASS) for a batch."""
    S_list = [
        semantic_similarity(pred, targets)
        for (pred, targets) in zip(preds, target_lists)
    ]
    detected_langs = detect_lang_batch(preds)
    L_list = [
        1.0 if l == target_l else 0.0
        for (l, target_l) in zip(detected_langs, target_langs)
    ]
    results = [beta * S + (1 - beta) * L for (S, L) in zip(S_list, L_list)]
    return results


def get_lass(dataframe: pd.DataFrame) -> List[str]:
    """Calculates the LASS metric for SFQA."""
    logger.info(f"Begin LASS calculation, len dataframe: {len(dataframe)}")
    scores = batch_lass(
        preds=dataframe["extracted_answer"],
        target_lists=dataframe["ground_truth"],
        target_langs=dataframe["answer_language"],
    )
    dataframe["LASS"] = scores
    return dataframe
