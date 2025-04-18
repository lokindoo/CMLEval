# General

This is a repository for the "Evaluating the cross- and multi-lingual capabilities of Large Language Models (LLM)" paper.

# Current progress

So far, I have outlined the main evaluation methods, and have chosen several methods which could be used to evaluate the LLMs. 

The main methods are:
- Multiple Choice Question Answering (MCQA)
- Correct Phrase Continuation (CPC)

Other open-ended generation like Slot Filling may be used, but more preparation is needed.
Generally, the methods will be combined with language augmentation techniques to create several versions of each task sample:
(MCQA)
- English question - X answers
- X question - English answers
- Mixed answers
- X correct answer

The main metrics to evaluate each method are: 
- MCQA - Precision, Accuracy, Cohen's Kappa 
- CoCo-CoLa, Cossim / Cross-lingual Consistency 

The evaluation prompt so far has included CoT based on the intuition that CoT generally improves results. Tests without CoT need to be conducted to prove this theory in the cross- multilingual context.

# Running evaluations

To run LLM evaluations, install requirements, create a config.yaml file based on the example file, and run the `evaluate_llm.py` script with the following command in the root directory:
```
python -m scripts.evaluation.evaluate_llm --dataset-path "path\to\dataset.parquet.gzip" --output-file "path\to\output.json" --config-path "path\to\config.yaml"
```
