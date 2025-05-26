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
EVALUATION_MCQA_SYS_PROMPT = (
    """You are an expert at answering multiple-choice questions."""
)

# 0-shot default prompt
EVALUATION_MCQA_PROMPT_0 = """Look at the question, think about it, and choose the correct answer from the ones given to you.
Write out your thought process, and in the end, put the correct answer as a latin uppercase letter in square brackets: [A], [B], etc.
One of these options is the correct answer.

Question: {question}
Options:
{options}
Answer:"""

# 1-shot using question no. 106 from kazakh dataset
EVALUATION_MCQA_PROMPT_1 = """Look at the question, think about it, and choose the correct answer from the ones given to you.
Write out your thought process, and in the end, put the correct answer as a latin uppercase letter in square brackets: [A], [B], etc.
One of these options is the correct answer.

Example:
Question: 1978 жылдан бастап деңгейі көтеріліп жатқан көл.
Options:
A. Bypass
B. Island
C. The Balk
D. Caspian
Answer: Let\'s break down this question step by step.\n\nThe question is in Kazakh, and it translates to "The lake that has been rising in level since 1978." \n\nNow, I need to think about which lake is known for its rising water level since 1978. \n\nHmm... I\'ve got it! The Caspian Sea is the world\'s largest inland body of water, and its water level has been rising since 1978 due to various natural and human factors. \n\nSo, the correct answer is [D]

Question: {question}
Options:
{options}
Answer:"""

# 2-shot using questions no. 106 & no. 179 from kazakh dataset
EVALUATION_MCQA_PROMPT_2 = """Look at the question, think about it, and choose the correct answer from the ones given to you.
Write out your thought process, and in the end, put the correct answer as a latin uppercase letter in square brackets: [A], [B], etc.
One of these options is the correct answer.

Example 1:
Question: 1978 жылдан бастап деңгейі көтеріліп жатқан көл.
Options:
A. Bypass
B. Island
C. The Balk
D. Caspian
Answer: Let\'s break down this question step by step.\n\nThe question is in Kazakh, and it translates to "The lake that has been rising in level since 1978." \n\nNow, I need to think about which lake is known for its rising water level since 1978. \n\nHmm... I\'ve got it! The Caspian Sea is the world\'s largest inland body of water, and its water level has been rising since 1978 due to various natural and human factors. \n\nSo, the correct answer is [D]

Example 2:
Question: Интернет арқылы қатысушылардың көп санынан аздаған ақша қаражатын жинау жолымен жобаны қаржыландыру практикасы қалай аталады
Options:
A. crowdfunding
B. start up
C. platform
D. marketing
Answer: A new question!\n\nLet's break it down step by step. The question is asking about a practice of financing a project by collecting small amounts of money from a large number of participants through the internet.\n\nHmm... This sounds familiar. I've heard of a concept that fits this description. It's a way of funding a project by raising money from a large number of people, typically through the internet.\n\nAh-ha! I think I have it! The correct answer is... [A]

Question: {question}
Options:
{options}
Answer:"""


EVALUATION_SFQA_SYS_PROMPT = (
    """You are an expert at answering short-answer questions."""
)

# default short answer prompt using 1 random English, 1 random Sundanese, and 1 random Arabic example
EVALUATION_SFQA_PROMPT = """Look at the question, think about it, and write a short answer to it.
Write out your thought process, and in the end, put the correct short answer in square brackets: [steak and fries], [كرة القدم], [taman umum], etc.
Make sure the short answer is in {answer_language} only.

Question: {question}
Answer:"""

EVALUATION_SYS_PROMPT_DICT = {
    "MCQA": EVALUATION_MCQA_SYS_PROMPT,
    "SFQA": EVALUATION_SFQA_SYS_PROMPT,
}

EVALUATION_MCQA_PROMPT_DICT = {
    "0": EVALUATION_MCQA_PROMPT_0,
    "1": EVALUATION_MCQA_PROMPT_1,
    "2": EVALUATION_MCQA_PROMPT_2,
}


def create_eval_prompt(row: pd.Series, prompt: str, qa_type: str) -> str:
    question = row["Question"]
    if qa_type == "MCQA":
        options = [
            row["Option A"],
            row["Option B"],
            row["Option C"],
            row["Option D"],
            row["Option E"],
        ]
        options = [o for o in options if o]
        options = "\n".join(
            [f"{chr(65 + i)}. {o}".strip() for i, o in enumerate(options)]
        )
        return prompt.format(question=question, options=options)
    else:
        if row["Direction"] == "en_answer":
            answer_language = "English"
        else:
            answer_language = row["Language"]
        return prompt.format(question=question, answer_language=answer_language)


EXTRACT_MCQA_PROMPT = """You are an expert at extracting a specific final answer from a long explanation. Analyse the explanation given, and select the final answer it comes to.
The answer must be a single capital Latin letter: A, B, C, D, E.
You must write only the answer, do not write anything else.
If no single Latin letter can be found as the answer, write "None".

Example:
Explanation: "...ythological", doesn\'t make sense in this context, as it\'s not a cellular component.\n\nOption B, "Lissoma", is not a familiar term in cell biology.\n\nOption C, "The center", is a bit vague, but it could be referring to the center of the cell, which is where the nucleus is located. However, the nucleus isn\'t typically associated with generating power for the cell.\n\nOption D, "Chloroplasts", is a type of organelle found in plant cells that is responsible for photosynthesis, which is the process of generating energy from light. This sounds like a good fit for the "Power station" description.\n\nOption E, "The kernel", is a term that\'s not commonly used in cell biology, and it doesn\'t seem to fit the description.\n\nBased on my analysis, I\'m going to choose option D, Chloroplasts.\n\n[Answer: D]"
FinalAnswer: D

Explanation: {explanation}
FinalAnswer:"""

EXTRACT_SFQA_PROMPT = """You are an expert at extracting a specific short answer from a long explanation. Analyse the explanation given, and select the final answer it comes to.
The answer must be a short text enclosed in square brackets [].
You must write only the answer text, do not write anything else.
If no answer can be found, write "None".

Example:
Explanation: "...y. A sweet pastry or croissant with a glass of milk or tea is also common in Algerian households.\n\nThese snacks are practical, quick to prepare, and accessible, aligning with typical breakfast or pre-school snack habits in Algerian culture.\n\nCorrect short answer: [bread with jam or chocolate spread]"
FinalAnswer: bread with jam or chocolate spread

Explanation: {explanation}
FinalAnswer:"""

EXTRACT_PROMPT_DICT = {
    "MCQA": EXTRACT_MCQA_PROMPT,
    "SFQA": EXTRACT_SFQA_PROMPT,
}
