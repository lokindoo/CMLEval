import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def load_checkpoint(output_file: str) -> Tuple:
    """Loads the checkpointed results_dict and last_processed_index if present."""
    output_file = output_file.split(".")[0]
    pkl = Path(output_file).with_suffix(".checkpoint.pickle")
    idx_file = Path(output_file).with_suffix(".checkpoint.idx")
    if pkl.exists() and idx_file.exists():
        with open(pkl, "rb") as h:
            results = pickle.load(h)
        with open(idx_file, "r") as f:
            last_idx = int(f.read().strip())
        return results, last_idx
    return None, 0


def save_checkpoint(results: Dict, output_file: str, last_idx: int):
    """Dump results and last index processed."""
    output_file = output_file.split(".")[0]
    pkl = Path(output_file).with_suffix(".checkpoint.pickle")
    idx_file = Path(output_file).with_suffix(".checkpoint.idx")
    with open(pkl, "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    with open(idx_file, "w") as f:
        f.write(str(last_idx))


def save_df(dataframe: pd.DataFrame, path: str):
    if path.endswith(".json"):
        dataframe.to_json(path, force_ascii=False, indent=2, orient="records")
    elif path.endswith(".parquet.gzip"):
        dataframe.to_parquet(path, index=False, compression="gzip")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
