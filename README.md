# ResMerge

ResMerge is a data-free merging method designed for reinforcement learning
post-trained language model experts that share the same base model. It targets
the spectral structure of reinforcement learning post-training task vectors by
decomposing each update into a rank-`K` spectral head and a residual tail,
building a stable residual backbone with SRC-A, and injecting the spectral head
only as a reliability-gated lightweight correction.

This repository provides the main merging script:

```text
resmerge-main.py
```

The script loads HuggingFace `safetensors` checkpoints in a streaming manner,
supports single-GPU and multi-GPU tensor-level parallelism, and saves the merged
model as a standard HuggingFace-compatible checkpoint.

## Method Overview

<p align="center">
  <img src="assets/overview.png" width="850">
</p>

Given a base model parameter matrix `W0` and `N` expert parameters `Wi`, ResMerge
first forms task vectors:

```text
Delta_i = W_i - W_0
```

For each eligible 2D weight matrix, ResMerge applies SVD and splits the task
vector into two components:

```text
Delta_i = H_i + R_i
```

- `H_i`: rank-`K` spectral head from the leading singular directions.
- `R_i`: residual tail after removing the spectral head.

The residual tails are merged by SRC-A. Let `q_src` denote the spherical
consensus direction estimated from normalized residual tails, `n_R` the average
residual norm, and `alpha_R` the confidence-dependent residual retention. SRC-A
constructs the residual backbone as:

```text
A = alpha_R * n_R * q_src
```

The spectral heads are averaged into a candidate head direction. ResMerge then
computes cross-expert head agreement `S_H` and injects only a lightweight
correction:

```text
C = rho * S_H * ||A|| * normalize(H_mean)
```

The final merged task vector is:

```text
M = A + C
W_merge = W_0 + M
```

For non-eligible tensors such as embeddings, `lm_head`, normalization tensors,
and 1D parameters, the default behavior is task arithmetic.

## Environment Setup

Recommended environment:

```bash
conda create -n resmerge python=3.10
conda activate resmerge
```

Install PyTorch matching your CUDA version, then install the minimal
dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Example command:

```bash
python resmerge-main.py \
  --base_model /path/to/base_model \
  --task_models /path/to/expert_1 /path/to/expert_2 /path/to/expert_3 \
  --output_dir /path/to/resmerge_output \
  --device cuda \
  --gpu_ids 0,1,2,3 \
  --merge_impl stream \
  --rank_topk 1 \
  --head_ratio_budget 0.20 \
  --head_reliability_rule pos_mean \
  --save_dtype bf16 \
  --save_attn_implementation eager \
  --retry_cuda_oom
```

The output directory will contain a standard HuggingFace model checkpoint:

```text
config.json
generation_config.json
model-*.safetensors
model.safetensors.index.json
tokenizer files
```

You can load it with:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/resmerge_output"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
```

## Arguments Used in Quick Start

```text
--base_model
    Path to the shared base model.

--task_models
    Paths to expert models fine-tuned from the same base model.

--output_dir
    Directory for the merged HuggingFace checkpoint.

--device
    Use cuda or cpu. CUDA is recommended.

--gpu_ids
    Comma-separated GPU ids used for tensor-level parallel merging.

--rank_topk
    Number of leading singular components treated as the spectral head.
    The default paper setting is 1.

--head_ratio_budget
    The residual-relative head budget rho. It controls the maximum norm of the
    lightweight head correction relative to the SRC-A residual backbone.

--head_reliability_rule
    Head agreement rule. `pos_mean` uses the positive mean pairwise cosine among
    expert heads as the reliability gate.

--save_dtype
    Output checkpoint dtype. The default paper setting is bf16.
```

## Evaluation

The merged checkpoint can be evaluated as a normal HuggingFace causal language
model. Following the evaluation setting used in
[MRL](https://github.com/xiangchi-yuan/mrl), we use the following tools:

- **Tool using:** [Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).
- **Memory:** we use [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent) to test long-context retention and retrieval capabilities.
- **Math and coding:** [Evalchemy](https://github.com/mlfoundations/evalchemy) as the unified evaluation framework.

Please keep the same prompt templates, decoding settings, dataset versions, and
sample counts when comparing the base model, expert models, baseline merging
methods, and ResMerge.
