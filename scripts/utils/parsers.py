import os
import re
from typing import Dict, Tuple

from dotenv import load_dotenv
from tqdm import tqdm
from scripts.utils.model_wrappers import company2wrapper
from scripts.utils.prompts import EXTRACT_PROMPT_DICT

load_dotenv()
EVAL_MODEL = os.getenv("EVAL_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def parse_checker_response(response: str) -> Tuple[str, str, str]:
    """Parse the response from the sanity checker model."""
    response = response.strip()
    llm_answer = None
    evaluation = None
    explanation = None

    # response structure
    """
    IndependentAnswer: The option letter you determine is correct
    Evaluation: YES if the marked answer is correct, NO if incorrect
    Explanation: Brief explanation of why
    """

    llm_answer, evaluation = response.split("Evaluation: ")
    llm_answer = llm_answer.strip().removeprefix("IndependentAnswer: ")
    evaluation, explanation = evaluation.split("Explanation: ")
    evaluation = evaluation.strip()
    explanation = explanation.strip()

    return llm_answer, evaluation, explanation


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

sfqa_patterns = [r"\[.+?\]"]


def parse_llm_answer(long_answer: str, patterns: list) -> str:
    """Parse a single LLM answer to extract the final answer."""
    part = 9 * len(long_answer) // 10
    long_answer = long_answer[part:]
    found = [re.search(p, long_answer) for p in patterns]
    found = [f[0] for f in found if f]
    final_answer = found[0].strip("[").strip("]") if found else ""

    return final_answer


# add qa_type
def extract_answers_with_rules(results: Dict, qa_type: str) -> Dict:
    if qa_type == "MCQA":
        patterns = mcqa_patterns
    else:
        patterns = sfqa_patterns
    for model in results.keys():
        for d in tqdm(results[model], total=len(results[model]), ncols=100):
            if not d["output"]:
                d["extracted_answer"] = ""
            else:
                if not d.get("extracted_answer"):
                    d["extracted_answer"] = parse_llm_answer(d["output"], patterns)


def extract_answers_with_llm(results: Dict, qa_type: str, test: bool) -> Dict:
    api_wrapper = company2wrapper.get("GROQ")
    extractor = api_wrapper(
        name=EVAL_MODEL,
        api_key=GROQ_API_KEY,
        qa_type=qa_type,
    )

    extract_prompt = EXTRACT_PROMPT_DICT[qa_type]

    for model in results.keys():
        for d in tqdm(results[model], total=len(results[model]), ncols=100):
            if not d["output"]:
                d["extracted_answer"] = ""
            else:
                if not d.get("extracted_answer"):
                    explanation = "..." + d["output"][len(d["output"]) - 300 :]
                    prompt = extract_prompt.format(explanation=explanation)
                    # Approximate buffer time based on Free Tier Groq API limits
                    # time.sleep(6)
                    extracted_answer = extractor.predict(prompt)
                    d["extracted_answer"] = extracted_answer.split("FinalAnswer:")[
                        -1
                    ].strip()
                    if test:
                        d["extracted_with_llm"] = True
                else:
                    if test:
                        d["extracted_with_llm"] = False
