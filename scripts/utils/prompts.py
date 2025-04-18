import pandas as pd

# Sanity check
SANITY_CHECK_PROMPT = """You are an expert reviewing answers to multiple-choice questions. You have a question and answer options, as well as a potentially correct answer. You need to determine if the marked answer is correct.
Question: {question}
Options:
{options}
The option {marked_answer} is potentially correct.

First, solve the question independently, then evaluate whether the marked answer (option {marked_answer}) is correct.
You must use this format:

IndependentAnswer: The option letter you determine is correct
Evaluation: YES if the marked answer is correct, NO if incorrect
Explanation: Brief explanation of why"""

# Evaluation
EVALUATION_SYS_PROMPT = """You are an expert multiple-choice question reviewer."""

EVALUATION_PROMPT = """You are an expert at testing multiple-choice questions. Look at the question, think about it, and choose the correct answer option from the ones given to you.
Write out your thought process, and in the end, put the answer in square brackets: [A], [B], etc.
One of these answers is correct.
Question: {question}
Options:
{options}
Answer:"""


def create_eval_prompt(row: pd.Series, prompt: str) -> str:
    question = row["Question"]
    options = [
        row["Option A"],
        row["Option B"],
        row["Option C"],
        row["Option D"],
        row["Option E"],
    ]
    options = [o for o in options if o]
    options = "\n".join([f"{chr(65 + i)}. {o}".strip() for i, o in enumerate(options)])
    return prompt.format(question=question, options=options)


# Extraction
EXTRACT_PROMPT = """You are an expert at extracting a specific final answer from a long explanation. Analyse the explanation given, and select the final answer it comes to.
The answer must be a single capital English letter: A, B, C, D, E.
You must write only the answer, do not write anything else.

Explanation: {explanation}
FinalAnswer:
"""
