from typing import Tuple


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
