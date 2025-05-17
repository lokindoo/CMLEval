import logging
import os
from typing import Optional

import click
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y/%m/%d %H:%M",
)
# TODO: put HTTP into warning level
logger = logging.getLogger(__name__)


def load_data(input_file: str) -> pd.DataFrame:
    if input_file.endswith(".csv"):
        return pd.read_csv(input_file)
    elif input_file.endswith(".parquet") or input_file.endswith(".gzip"):
        return pd.read_parquet(input_file)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a CSV or Parquet file."
        )


@click.command()
@click.option("--input-file", help="Path to the input CSV or Parquet file.")
@click.option("--output-file", help="Path to the output CSV or Parquet file.")
@click.option("--local/--remote", default=True, help="Mode of translation.")
def main(input_file: str, output_file: str, local: Optional[bool]):
    df = load_data(input_file)
    logger.info("Dataset loaded")

    if local:
        logger.info("Preparing local translations")
        # model_name = "facebook/nllb-200-distilled-600M"
        model_name = "facebook/nllb-200-distilled-1.3B"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, local_files_only=True
        )  # , cache_dir=r"C:\Users\nikol\OneDrive\Desktop\project_repos\CMLEval\models")

        # TODO: add more languages
        en = "eng_Latn"
        ru = "rus_Cyrl"
        kaz = "kaz_Cyrl"

        tokenizers = {
            "Russian": AutoTokenizer.from_pretrained(
                model_name,
                src_lang=ru,
                tgt_lang=en,
            ),
            "Kazakh": AutoTokenizer.from_pretrained(
                model_name,
                src_lang=kaz,
                tgt_lang=en,
            ),
        }

        def translate_text(text: str, src_lang: str) -> str:
            try:
                tokenizer = tokenizers[src_lang]
                model_inputs = tokenizer(text, return_tensors="pt")
                gen_tokens = model.generate(
                    **model_inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(en),
                    max_length=len(text) + 50,
                )
                result = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                return result[0]
            except Exception as e:
                print(f"Error translating text: {text}\nError: {e}")
                return ""

    else:
        # TODO: add batch translation with gpt
        logger.info("Preparing remote translations")
        model = "gpt-4o-mini"
        client = OpenAI(api_key=OPENAI_KEY)

        def translate_text(text: str, src_lang: str) -> str:
            prompt = (
                f"Translate a text from {src_lang} to English, include only the direct translation in your answer, do not write anything else or answer any questions.\nText:\n"
                + text
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert translator in English and {src_lang}.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                service_tier="flex",
            )
            return response.choices[0].message.content

    to_translate = [
        "Option A",
        "Option B",
        "Option C",
        "Option D",
        "Option E",
    ]
    tqdm.pandas(ncols=50)
    translations = []
    for _, row in tqdm(
        df.iterrows(), total=len(df), ncols=50, desc="Translating dataset to English"
    ):
        parts = [row[c] for c in to_translate]
        parts = [e for e in parts if e]
        parts = [f"{i+1}. {e.strip()}" for i, e in enumerate(parts)]
        text = " ".join(parts)
        text = row["Question"].strip().strip(":") + ": " + text
        translated = translate_text(text=text, src_lang=row["Language"])
        translations.append(translated)
    translations = pd.DataFrame(translations, columns=["en_translation"])
    df = df.reset_index()
    df = pd.concat([df, translations], axis=1).copy()

    logger.info(f"Saving dataset to {output_file}")
    if output_file.endswith(".csv"):
        df.to_csv(output_file, index=False)
    elif output_file.endswith(".parquet") or output_file.endswith(".gzip"):
        df.to_parquet(output_file, compression="gzip")
    else:
        raise ValueError("Unsupported output file format. Please use CSV or Parquet.")


if __name__ == "__main__":
    main()
