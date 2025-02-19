# General

This is a repository for the "Evaluating the cross- and multi-lingual capabilities of Large Language Models (LLM)" paper.

# Current progress

So far, I have outlined the main evaluation methods, and have chosen several datasets which could be used to evaluate the LLMs. 

The datasets, in no particular order, are:
1. Mr. TyDi - `https://github.com/castorini/mr.tydi?tab=readme-ov-file`
2. CulturaX - `https://arxiv.org/abs/2309.09400`
3. Aya dataset - `https://arxiv.org/abs/2402.06619`

# TODO

1. Test out translation of current datasets in preparation for cross-lingual capability testing
    - test out NLLB / OPUS models for translating some samples
    - test the translation quality via free LLM-as-a-judge candidates (groq?)
    - evaluate actual translation quality and choose the best judge
2. Translate a representative portion of the chosen datasets and assemble into first benchmark version
    - human validation
3. TBD
