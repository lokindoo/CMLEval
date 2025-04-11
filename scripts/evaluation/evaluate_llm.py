import json
import logging
import os
import time
from typing import Optional

import click
import pandas as pd
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

from ..utils.model_wrappers import LocalLLM, company2wrapper
from ..utils.prompts import EVALUATION_PROMPT, create_eval_prompt

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y/%m/%d %H:%M",
)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def evaluate_models(dataset, models):
    labels = [
        "marked_answer",
    ]
    ground_truth_label = [e for e in labels if e in dataset.columns][0]

    results = {m.name: [] for m in models}
    for i, sample in tqdm(
        dataset.iterrows(),
        desc="Testing chosen models...",
        ncols=100,
    ):
        # Buffer time based on Groq API limits
        if (i + 1) % 12 == 0:
            time.sleep(60)
        try:
            prompt = create_eval_prompt(sample, EVALUATION_PROMPT)
            for model in models:
                output = model.predict(prompt)
                results[model.name].append(
                    {
                        "prompt": prompt,
                        "output": output,
                        "ground_truth": sample[ground_truth_label],
                    }
                )
        except Exception as e:
            print(f"Error: {e}")
            for model in models:
                results[model.name].append(None)
    return results


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
    "--config-path", required=True, help="YAML file containing model parameters."
)
@click.option(
    "--dataset-interval",
    help="Comma-separated indices for beginning and ending of the desired dataset interval.",
)
def main(
    dataset_path: str,
    output_file: str,
    config_path: str,
    dataset_interval: Optional[str],
):

    with open(config_path, "r") as file:
        stream = file.read()
    config = yaml.safe_load(stream)
    if not config:
        raise Exception(
            "No models indicated. Check the config file and re-run script with at least 1 model."
        )

    models = []
    for model_dict in tqdm(
        config,
        desc="Config file found, loading models...",
        ncols=100,
    ):
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
            api_wrapper = company2wrapper.get(company) or "DEFAULT"
            models.append(
                api_wrapper(
                    name=model_dict.get("name"),
                    api_key=eval(api_key),
                )
            )

    dataset = pd.read_parquet(dataset_path)
    if dataset_interval:
        s, e = dataset_interval.split(",")
        dataset = dataset[int(s) : int(e)].copy()
    results = evaluate_models(dataset, models)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
