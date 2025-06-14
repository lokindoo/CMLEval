import os
import re
from string import punctuation
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from utils.model_wrappers import company2wrapper
from utils.prompts import EXTRACT_PROMPT_DICT

load_dotenv()
EVAL_MODEL = os.getenv("EVAL_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

mcqa_patterns = [
    r"\[[A-E]\]",
    r"[A-E](?=\.)",
    r"(?!=\*{2})[A-E](?=\*{2})",
    r"(?<=\[Answer: )[A-E]",
    r"(?<=\[Final Answer: )[A-E]",
    r"(?<=\[Correct Answer: )[A-E]",
    r"(?<=The answer is: )[A-E]",
    r"(?<=The answer is )[A-E]",
    r"(?<=option )[A-E]",
    r"(?<=Option )[A-E]",
]

sfqa_patterns = [
    r"\[.+?\]",
    r"""(?<=[aA]nswer in Arabic: ).+[\]'"»\.]?""",
    r"""(?<=[aA]nswer in Assamese: ).+[\]'"»\.]?""",
    r"""(?<=[aA]nswer in Azerbaijani: ).+[\]'"»\.]?""",
    r"""(?<=[aA]nswer in Amharic: ).+[\]'"»\.]?""",
    r"""(?<=[aA]nswer in Amharic: ).+[\]'"»\.]?""",
    r"""(?<=[aA]nswer in Greek: ).+[\]'"»\.]?""",
    r"""(?<=[aA]nswer in Persian: ).+[\]'"»\.]?""",
    r"""(?<=[aA]nswer in Korean: ).+[\]'"»\.]?""",
    r"""(?<=[aA]nswer in Hausa: ).+[\]'"»\.]?""",
    r"""(?<=[aA]nswer in Sundanese: ).+[\]'"»\.]?""",
    r"""(?<=[aA]nswer in English: ).+[\]'"»\.]?""",
]


def parse_llm_answer(long_answer: str, patterns: list, qa_type: str) -> str:
    """Parse a single LLM answer to extract the final answer."""
    if qa_type == "MCQA":
        long_answer = long_answer[-300:]
    found = [re.search(p, long_answer) for p in patterns]
    found = [f[0] for f in found if f]
    final_answer = found[-1].strip().strip(punctuation) if found else ""

    return final_answer


def extract_answers_with_rules(
    results: pd.DataFrame, qa_type: str, force: Optional[bool] = False
) -> pd.DataFrame:
    if qa_type == "MCQA":
        patterns = mcqa_patterns
    else:
        patterns = sfqa_patterns
    extracted_answers = []
    for _, row in tqdm(results.iterrows(), total=len(results), ncols=100):
        if not row["output"]:
            extracted_answers.append("")
        else:
            if not row.get("extracted_answer") or force:
                extracted_answers.append(
                    parse_llm_answer(row["output"], patterns, qa_type)
                )
            else:
                extracted_answers.append(row.get("extracted_answer"))
    results["extracted_answer"] = extracted_answers

    return results


def extract_answers_with_llm(
    results: pd.DataFrame, qa_type: str, test: bool
) -> pd.DataFrame:
    api_wrapper = company2wrapper.get("GROQ")
    extractor = api_wrapper(
        name=EVAL_MODEL,
        api_key=GROQ_API_KEY,
        qa_type=qa_type,
        extraction=True,
    )

    extract_prompt = EXTRACT_PROMPT_DICT[qa_type]
    extracted_answers = []
    with_llm = []
    for _, row in tqdm(results.iterrows(), total=len(results), ncols=100):
        if not row["output"]:
            extracted_answers.append("")
            if test:
                with_llm.append(False)
        else:
            if a := row.get("extracted_answer"):
                extracted_answers.append(a)
                if test:
                    with_llm.append(False)
            else:
                explanation = "..." + row["output"][-300:]
                prompt = extract_prompt.format(explanation=explanation)
                # Approximate buffer time based on Free Tier Groq API limits
                # time.sleep(6)
                e = extractor.predict(prompt)
                extracted_answers.append(e.split("FinalAnswer:")[-1].strip())
                if test:
                    with_llm.append(True)
    results["extracted_answer"] = extracted_answers
    if test:
        results["extracted_with_llm"] = with_llm

    return results
