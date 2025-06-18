from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="LLM Evaluation Dashboard", layout="wide")
st.title("LLM Evaluation Dashboard")

results_dir = Path(".") / "results"
files = list(results_dir.glob("**/*.parquet.gzip"))
df = []
for filepath in files:
    d = pd.read_parquet(filepath)
    d["qa_type"] = filepath.parts[-2]
    df.append(d)
df = pd.concat(df, ignore_index=True)
df["Language"] = df.apply(
    lambda r: (
        r["question_language"]
        if r["question_language"] != "English"
        else r["answer_language"]
    ),
    axis=1,
)

st.sidebar.header("Filters & Metrics")
models = df["model"].unique().tolist()
languages = df["Language"].unique().tolist()

selected_model = st.sidebar.selectbox("Select model", ["All"] + models)
selected_languages = st.sidebar.multiselect(
    "Select languages", languages, default=languages
)

cols = ["Model"]
for lang in df["Language"].unique():
    if lang != "English":
        cols.append(lang)

data = []
for model in df["model"].unique():
    line_content = [model]
    subset = df[df["model"] == model].copy()
    for language in subset["Language"].unique():
        if language not in ["Basque", "Kazakh"]:
            line_content.append(
                round(100 * subset[subset["Language"] == language]["LASS"].mean(), 2)
            )
        else:
            lang_subset = subset[subset["Language"] == language]
            line_content.append(
                round(
                    100
                    * len(
                        lang_subset[
                            lang_subset["extracted_answer"]
                            == lang_subset["ground_truth"]
                        ]
                    )
                    / len(lang_subset),
                    2,
                )
            )
    data.append(line_content)
model_language_df = pd.DataFrame(
    data,
    columns=cols,
)

filtered = model_language_df.copy()
if selected_model != "All":
    filtered = filtered[filtered["Model"] == selected_model]
if selected_languages:
    filtered = filtered[["Model"] + selected_languages]

if selected_model != "All":
    st.subheader(f"Performance score by language for {selected_model}")

    lang_acc = filtered.T.reset_index()[1:]
    lang_acc.columns = ["Language", "Performance Score (LASS or Exact Match)"]
    lang_acc = lang_acc.sort_values(
        by="Performance Score (LASS or Exact Match)", ascending=False
    )

    fig_lang = px.bar(
        lang_acc,
        x="Language",
        y="Performance Score (LASS or Exact Match)",
        title="Performance score by language",
    )
    st.plotly_chart(fig_lang, use_container_width=True)

st.subheader("Model Comparison")
cols = ["Model", "Mean performance Score (LASS or Exact Match)"]

data = []
for model in df["model"].unique():
    line_content = [model]
    subset = df[df["model"] == model].copy()
    scores = []
    for language in subset["Language"].unique():
        if language not in ["Basque", "Kazakh"]:
            scores.append(
                round(100 * subset[subset["Language"] == language]["LASS"].mean(), 2)
            )
        else:
            lang_subset = subset[subset["Language"] == language]
            scores.append(
                round(
                    100
                    * len(
                        lang_subset[
                            lang_subset["extracted_answer"]
                            == lang_subset["ground_truth"]
                        ]
                    )
                    / len(lang_subset),
                    2,
                )
            )
    line_content.append(round(sum(scores) / len(scores), 2))
    data.append(line_content)
model_acc = pd.DataFrame(
    data,
    columns=cols,
)
model_acc = model_acc.sort_values(
    "Mean performance Score (LASS or Exact Match)", ascending=False
)
fig_model = px.bar(
    model_acc,
    x="Model",
    y="Mean performance Score (LASS or Exact Match)",
    title="Model Accuracy Comparison",
)
st.plotly_chart(fig_model, use_container_width=True)

st.subheader("Accuracy Heatmap (Model vs Language)")
heat_df = model_language_df[selected_languages].set_index(model_language_df["Model"])
heat_df["avg"] = heat_df.apply(
    lambda r: r[selected_languages].mean(),
    axis=1,
)
heat_df = heat_df.sort_values(by="avg", ascending=False).drop(columns=["avg"])
fig_heat = px.imshow(
    heat_df,
    labels=dict(
        x="Language", y="Model", color="Performance Score (LASS or Exact Match)"
    ),
    text_auto=True,
    color_continuous_scale="RdBu_r",
)
st.plotly_chart(fig_heat, use_container_width=True)

with st.expander("Show Raw Data"):
    st.dataframe(filtered[:10000], use_container_width=True)

st.markdown("---")
