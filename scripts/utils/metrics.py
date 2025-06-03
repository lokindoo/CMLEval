import logging
import time
from typing import List

import fasttext
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

logging.basicConfig(
    filename="metrics.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%d-%m %H:%M:%S",
)
logger = logging.getLogger(__name__)


detector = fasttext.load_model(
    r"C:\Users\nikol\OneDrive\Desktop\project_repos\CMLEval\scripts\utils\lid.176.bin"
)

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
        result = detector.predict(t)
        lang_code = result[0][0].replace("__label__", "")
        langs.append(code2lang.get(lang_code, lang_code))
    return langs


def semantic_similarity(pred: str, targets: list[str]) -> float:
    embeddings = embedder.encode(
        [pred] + targets.tolist(), convert_to_tensor=True, show_progress_bar=False
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
    st = time.time()
    S_list = [
        semantic_similarity(pred, targets)
        for (pred, targets) in zip(preds, target_lists)
    ]
    et = time.time()
    logging.info(f"Semantic sims, time taken: {et - st}s")
    detected_langs = detect_lang_batch(preds)
    et2 = time.time()
    logging.info(f"Lang detect, time taken: {et2 - et}s")
    L_list = [
        1.0 if l == target_l else 0.0
        for (l, target_l) in zip(detected_langs, target_langs)
    ]
    results = [beta * S + (1 - beta) * L for (S, L) in zip(S_list, L_list)]
    et = time.time()
    logging.info(f"Batch lass, total time taken: {et - st}s")
    return S_list, detected_langs, L_list, results


def get_lass_by_direction(dataframe: pd.DataFrame) -> List[str]:
    """Calculates LASS for two directions for SFQA."""
    # report = []
    results = {}
    for direction in dataframe["Direction"].unique():
        # report.append(f"{direction}\n")
        logging.info(f"{direction}\n")
        subset = dataframe[dataframe["Direction"] == direction].copy()
        for model in tqdm(
            subset["model"].unique(),
            total=len(subset["model"].unique()),
            desc="Processing per-model LASS",
            ncols=100,
        ):
            # TODO: remove
            if model != "gemma2-9b-it":
                logging.info(f"{model}\n")
                subset_2 = subset[subset["model"] == model].copy()
                if direction == "en_question":
                    S_list, detected_langs, L_list, scores = batch_lass(
                        preds=subset_2["extracted_answer"],
                        target_lists=subset_2["ground_truth"],
                        target_langs=subset_2["Language"],
                    )
                else:
                    S_list, detected_langs, L_list, scores = batch_lass(
                        preds=subset_2["extracted_answer"],
                        target_lists=subset_2["ground_truth"],
                        target_langs=["English"] * len(subset_2),
                    )
                # report.append(f"{model}: {round(100 * sum(lass_batch) / len(lass_batch), 2)}")
                lass_batch = zip(S_list, detected_langs, L_list, scores)
                results[f"{direction}-{model}"] = [
                    (i, batch[0], batch[1], batch[2], batch[3])
                    for i, batch in zip(subset_2.index, lass_batch)
                ]
        # report.append("\n\n")
    # return report
    return results
