# Evaluation

The merged checkpoint produced by `resmerge-main.py` is a standard
HuggingFace-compatible causal language model. We evaluate it with the same
protocol used for the base model, expert models, and baseline merged models.

## Tool Using

We use the Berkeley Function Calling Leaderboard (BFCL):

- Repository: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard

Follow the official BFCL setup and keep the same prompt templates, decoding
settings, and split configuration across all compared models. In the paper
setting, tool-use performance is reported over BFCL live and non-live function
calling tasks.

## Memory

We use MemAgent to test long-context retention and retrieval capabilities:

- Repository: https://github.com/BytedTsinghua-SIA/MemAgent

Follow the official MemAgent benchmark setup. In the paper setting, memory
performance is reported on long-context retrieval settings such as 7K, 14K, and
32K contexts.

## Math and Coding

We use Evalchemy as the unified evaluation framework for math and coding:

- Repository: https://github.com/mlfoundations/evalchemy

In our experiments, the math evaluation includes AIME24, AIME25, AMC23, and
MATH500. The coding evaluation includes LiveCodeBench, HumanEvalPlus, and
MBPPPlus. Please use the corresponding task identifiers and evaluation commands
from your local Evalchemy setup.

## Reporting

For fair comparison, keep the following settings identical across all models:

- prompt templates
- decoding parameters
- benchmark versions
- number of evaluated examples
- random seeds, if sampling is used

In the paper, we report both per-benchmark scores and aggregated domain scores.
