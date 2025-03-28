import argparse

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_data(input_file: str) -> pd.DataFrame:
    if input_file.endswith(".csv"):
        return pd.read_csv(input_file)
    elif input_file.endswith(".parquet") or input_file.endswith(".gzip"):
        return pd.read_parquet(input_file)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a CSV or Parquet file."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Translate text in a CSV/Parquet file using a HuggingFace model"
    )
    parser.add_argument("--infile", help="Path to the input CSV or Parquet file")
    parser.add_argument("--outfile", help="Path to the output CSV or Parquet file")
    args = parser.parse_args()

    df = load_data(args.infile)

    model_name = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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

    to_translate = [
        "question",
        "Option A",
        "Option B",
        "Option C",
        "Option D",
        "Option E",
    ]
    tqdm.pandas(ncols=50)
    for c in to_translate:
        print(f"Translating column {c}")
        df[f"en_{c}"] = df.progress_apply(
            lambda r: translate_text(text=r[c], src_lang=r["Language"]),
            axis=1,
        )

    if args.outfile.endswith(".csv"):
        df.to_csv(args.outfile)
    elif args.outfile.endswith(".parquet") or args.outfile.endswith(".gzip"):
        df.to_parquet(args.outfile, compression="gzip")
    else:
        raise ValueError("Unsupported output file format. Please use CSV or Parquet.")


if __name__ == "__main__":
    main()
