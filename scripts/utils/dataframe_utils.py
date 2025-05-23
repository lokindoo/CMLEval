import pandas as pd


def dict_to_df(input_dict: dict) -> pd.DataFrame:
    dfs = []
    for model in input_dict.keys():
        df = pd.DataFrame(input_dict[model])
        df["model"] = model
        dfs.append(df)
    return pd.concat(dfs)
