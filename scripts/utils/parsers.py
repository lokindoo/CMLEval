import re
from typing import Dict, List, Tuple


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


patterns = [
    r"\[[A-E]\]",
    r"(?<=\[Answer: )[A-E]",
    r"(?<=\[Final Answer: )[A-E]",
    r"(?<=\[Correct Answer: )[A-E]",
    r"[A-E](?=\.)",
    r"(?!=\*{2})[A-E](?=\*{2})",
    r"(?<=The answer is: )[A-E]",
    r"(?<=The answer is )[A-E]",
    r"(?<=option )[A-E]",
    r"(?<=Option )[A-E]",
]


def parse_llm_answer(long_answer: str) -> str:
    """Parse a single LLM answer to extract the final answer."""
    # TODO: use not_found to single out answers to extract with an LLM if option is set
    # uncomment
    # not_found = []
    part = 9 * len(long_answer) // 10
    long_answer = long_answer[part:]
    found = [re.search(p, long_answer) for p in patterns]
    found = [f[0] for f in found if f]
    # uncomment
    # if not found:
    #     not_found.append(e["output"])
    final_answer = found[0].strip("[").strip("]") if found else ""

    return final_answer
