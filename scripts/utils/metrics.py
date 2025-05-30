from typing import List

from fasttext import load_model  # or langid, cld3, etc.
from sentence_transformers import SentenceTransformer, util

lang_clf = load_model("lid.176.ftz")


def detect_lang(txt: str) -> str:
    """Helper function to detect language."""
    lang, prob = lang_clf.predict(txt.replace("\n", " "), k=1)
    return lang[0].replace("__label__", "")


embedder = SentenceTransformer("sentence-transformers/LaBSE")


def sem_score(pred: str, gold_list: List[str]) -> float:
    """Calculates semantic score and chooses highest."""
    embs = embedder.encode([pred] + gold_list, convert_to_tensor=True)
    sim = util.cos_sim(embs[0], embs[1:]).max().item()
    return max(sim, 0.0)


def lass(pred: str, gold_list: List[str], target_lang: str, beta: float = 0.8) -> float:
    """Calculates Language Aware Semantic Score (LASS)"""
    S = sem_score(pred, gold_list)
    # try using direct fasttext pred score
    L = 1.0 if detect_lang(pred) == target_lang else 0.0
    return beta * S + (1 - beta) * L
