# General

This is a repository for the "Evaluating the cross- and multi-lingual capabilities of Large Language Models (LLM)" paper.

# Setup

Before evaluating LLMs, create a virtual environment and run `pip install -r requirements.txt`.
Create config.yaml and .env files based on the provided example files.

# Running evaluations

To run LLM evaluations use the `evaluate_llm.py` script with the following command from the root directory:
```
python -m scripts.evaluation.evaluate_llm --config-path "path\to\config.yaml"
```

If needed, zero-shot prompting can be added, and the automatic LLM answer parsing module can be disconnected by adding the corresponding flags to the run command.

# Visualization

To view the LLM evaluation results in a convenient format, run the following command from the root directory:
```
streamlit run scripts\visualisation\dashboard.py
```

The streamlit app can be used to get a general impression of the metrics, as well as to compare how some models did against others, and see how each model performed in each language separately.
