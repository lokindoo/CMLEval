import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import click
import pandas as pd
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

from scripts.utils.dataframe_utils import dict_to_df
from scripts.utils.io import (
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_df,
)
from scripts.utils.metrics import get_exact_match, get_lass
from scripts.utils.model_wrappers import BaseLLM, LocalLLM, company2wrapper
from scripts.utils.parsers import extract_answers_with_llm, extract_answers_with_rules
from scripts.utils.prompts import (
    EVALUATION_MCQA_PROMPT_DICT,
    EVALUATION_SFQA_PROMPT,
    create_eval_prompt,
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


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
    qa_type: str,
    output_file: str,
    start_index: int,
) -> None:
    if qa_type == "MCQA":
        eval_prompt = EVALUATION_MCQA_PROMPT_DICT[fewshot]
    else:
        eval_prompt = EVALUATION_SFQA_PROMPT
    logger.info(f"FEWSHOT | {fewshot}-shot")
    logger.info(f"PROMPT |\n{eval_prompt}")

    for i, sample in tqdm(
        dataset.iterrows(),
        desc="Testing chosen models...",
        total=len(dataset),
        ncols=100,
    ):
        # Approximate buffer time based on Free Tier Groq API limits
        # time.sleep(6)
        prompt = create_eval_prompt(sample, eval_prompt, qa_type)
        for model in models:
            output = ""
            for attempt in range(3):
                try:
                    output = model.predict(prompt)
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.info(
                            f"ERROR | {i} | {model.name}: {e}. Retry attempt {attempt+1}"
                        )
                        time.sleep(1)
                        continue
                    else:
                        logger.info(f"FAILED | {i} | {model.name}: {e}")
            results[model.name].append(
                {
                    "output": output,
                    "ground_truth": sample["ground_truth"],
                    "question_language": sample["question_language"],
                    "answer_language": sample["answer_language"],
                    "prompt": prompt,
                }
            )

        if i != 0 and i % 10 == 0:
            save_checkpoint(results, output_file, start_index + i + 1)


@click.command()
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
    config_path: str,
    llm_answer_extract: Optional[bool],
    fewshot: str,
    test: bool,
):
    with open(config_path, "r") as file:
        stream = file.read()
    config = yaml.safe_load(stream)
    if not config:
        raise Exception(
            "No models indicated. Check the config file and re-run script with at least 1 model."
        )

    if [m for m in config if m.get("model_cache_path")]:
        login(os.getenv("HF_TOKEN"))
    models = []
    for model_dict in config:
        logger.info(f"LOADING | {model_dict.get('name')}")
        if cache_folder := model_dict.get("model_cache_path"):
            models.append(
                LocalLLM(
                    full_name=model_dict.get("name"),
                    cache_path=Path(*Path(".").absolute().parts[:-2] + (cache_folder,)),
                    qa_type=qa_type,
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
                    qa_type=qa_type,
                )
            )

    path = Path(*Path(".").absolute().parts[:-2] + ("data",))
    for filepath in path.glob("**/*.parquet"):
        qa_type = filepath.parts[-2]
        logger.info(f"DATASET | {filepath.name.split(".")[0]} | {qa_type}")
        output_file = Path(
            *Path(".").absolute().parts[:-2] + ("results", qa_type, filepath.name)
        )
        results, start_idx = load_checkpoint(output_file)
        if results is None:
            results = {m.name: [] for m in models}
        else:
            logger.info(f"RESUMING FROM | {start_idx}")

        df = pd.read_parquet(filepath)
        df = df.iloc[start_idx:].reset_index(drop=True)

        evaluate_models(df, models, results, fewshot, qa_type, output_file, start_idx)
        logger.info("Extracting final option from LLM output using rules.")
        results = dict_to_df(results)
        results = extract_answers_with_rules(results, qa_type)

        save_df(
            results,
            output_file,
        )

        if llm_answer_extract:
            logger.info("Extracting final option from LLM output using LLM.")
            results = extract_answers_with_llm(results, qa_type, test)

            save_df(
                results,
                output_file,
            )

        # TODO: add metric calculations based on qa_type

        if qa_type == "MCQA":
            results = get_exact_match(results)
        elif qa_type == "SFQA":
            results = get_lass(results)

        save_df(
            results,
            output_file,
        )

        delete_checkpoint(output_file)


if __name__ == "__main__":
    main()
