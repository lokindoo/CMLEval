import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import click
import pandas as pd
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

from ..utils.checkpointing import load_checkpoint, save_checkpoint
from ..utils.model_wrappers import BaseLLM, LocalLLM, company2wrapper
from ..utils.parsers import parse_llm_answer
from ..utils.prompts import EVALUATION_PROMPT_DICT, EXTRACT_PROMPT, create_eval_prompt

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")
EVAL_MODEL = os.getenv("EVAL_MODEL")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y/%m/%d %H:%M",
)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def evaluate_models(
    dataset: pd.DataFrame,
    models: List[BaseLLM],
    results: Dict,
    fewshot: str,
    output_file: str,
    start_index: int,
) -> Dict:
    eval_prompt = EVALUATION_PROMPT_DICT[fewshot]
    logger.info(f"Doing {fewshot}-shot eval. Using prompt:\n{eval_prompt}")

    for i, sample in tqdm(
        dataset.iterrows(),
        desc="Testing chosen models...",
        total=len(dataset),
        ncols=100,
    ):
        # Approximate buffer time based on Groq API limits
        # time.sleep(6)
        prompt = create_eval_prompt(sample, eval_prompt)
        for model in models:
            try:
                output = model.predict(prompt)
                results[model.name].append(
                    {
                        "prompt": prompt,
                        "output": output,
                        "ground_truth": sample["marked_answer"],
                    }
                )
            except Exception as e:
                print(f"Error with model {model.name}: {e}")
                results[model.name].append(
                    {
                        "prompt": prompt,
                        "output": "",
                        "ground_truth": "",
                    }
                )
        if i != 0 and i % 10 == 0:
            save_checkpoint(results, output_file, start_index + i + 1)


def extract_answers_with_rules(results: Dict) -> Dict:
    logger.info("Extracting final option from LLM output using rules.")
    for model in results.keys():
        logger.info(f"Model {model}\n")
        for d in tqdm(results[model], total=len(results[model]), ncols=100):
            if d["output"]:
                d["extracted_answer"] = parse_llm_answer(long_answer=d["output"])


def extract_answers_with_llm(results: Dict, test: bool) -> Dict:
    logger.info("Extracting final option from LLM output using LLM.")
    api_wrapper = company2wrapper.get("GROQ")
    extractor = api_wrapper(
        name=EVAL_MODEL,
        api_key=GROQ_KEY,
    )

    for model in results.keys():
        logger.info(f"Model {model}")
        for d in tqdm(results[model], total=len(results[model]), ncols=100):
            if not d["output"]:
                d["extracted_answer"] = ""
            else:
                if not d["extracted_answer"]:
                    # Approximate buffer time based on Groq API limits
                    # time.sleep(6)
                    explanation = "..." + d["output"][len(d["output"]) - 300 :]
                    prompt = EXTRACT_PROMPT.format(explanation=explanation)
                    extracted_answer = extractor.predict(prompt)
                    d["extracted_answer"] = extracted_answer.split("FinalAnswer:")[
                        -1
                    ].strip()
                    if test:
                        d["extracted_with_llm"] = True
                else:
                    if test:
                        d["extracted_with_llm"] = False


@click.command()
@click.option(
    "--dataset-path",
    required=True,
    help="Parquet file containing the dataset to be used for testing.",
)
@click.option(
    "--output-file", required=True, help="JSON file to save model evaluation results."
)
@click.option(
    "--config-path",
    required=True,
    default="example_config.yaml",
    help="YAML file containing model parameters.",
)
@click.option(
    "--llm-answer-extract/--manual-answer-extract",
    default=True,
    help="Use calls to another LLM to extract the final choice of the LLMs you are evaluating.",
)
@click.option("--fewshot", type=click.Choice(["0", "1", "2"]), default="0")
@click.option(
    "--test", is_flag=True, default=False, help="Enable developer testing capabilities."
)
def main(
    dataset_path: str,
    output_file: str,
    config_path: str,
    llm_answer_extract: Optional[bool],
    fewshot: str,
    test: bool,
):
    logger.info(f"Using dataset {Path(dataset_path).stem}")

    with open(config_path, "r") as file:
        stream = file.read()
    config = yaml.safe_load(stream)
    if not config:
        raise Exception(
            "No models indicated. Check the config file and re-run script with at least 1 model."
        )

    models = []
    for model_dict in config:
        logger.info(f"Loading model {model_dict.get('name')}")
        if cache_path := model_dict.get("model_cache_path"):
            login()
            models.append(
                LocalLLM(
                    full_name=model_dict.get("name"),
                    cache_path=cache_path,
                )
            )
        else:
            api_key = model_dict.get("api_key")
            company = api_key.split("_")[0]
            api_wrapper = company2wrapper.get(company) or company2wrapper.get("DEFAULT")
            models.append(
                api_wrapper(
                    name=model_dict.get("name"),
                    api_key=eval(api_key),
                )
            )

    # TODO: add interaction with several datasets at once
    results, start_idx = load_checkpoint(output_file)
    if results is None:
        results = {m.name: [] for m in models}
    else:
        logger.info(f"Resuming processing from record {start_idx}")

    df = pd.read_parquet(dataset_path)
    df = df.iloc[start_idx:].reset_index(drop=True)

    evaluate_models(df, models, results, fewshot, output_file, start_idx)
    extract_answers_with_rules(results)

    # with open(output_file, "r", encoding="utf-8") as f:
    #     results = json.load(f)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if llm_answer_extract:
        extract_answers_with_llm(results, test)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # TODO: delete idx and checkpoint after everything is done and saved


if __name__ == "__main__":
    main()
