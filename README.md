# CastReasoner

CastReasoner is an evidence-grounded framework for context-aware time series forecasting. It analyzes observed history and available future covariates, then chooses to retain, revise, or reconstruct a forecast prior to produce the final forecast. Training consists of supervised fine-tuning (SFT) followed by GRPO reinforcement learning.

## Environment setup

Use Python 3.10 or newer with CUDA-enabled PyTorch. The default training configuration targets four H100 80GB GPUs.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 vllm==0.18.1 packaging ninja
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e '.[train]'
```

Training uses VeRL and vLLM. Cache the required text encoder before running GRPO:

```bash
hf download BAAI/bge-m3 --local-dir models/bge-m3
export CASTREASONER_ENCODER_PATH="$PWD/models/bge-m3"
```

## Running CastReasoner

Provide your own model and prepared Parquet training files compatible with [the dataset adapters](castreasoner/datasets.py), or prepare the Price example below. Prepared training data and model weights are not included. The main model is Qwen3.5-9B; Qwen3.5-4B uses the same commands with a different model path.

A runnable [Price example](examples/price/README.md) includes raw CSV observations, CSV-to-Parquet preparation using four configurable teacher APIs, and a single-command SFT → GRPO → inference → evaluation script. After installing `.[train,example]` and configuring the teacher APIs and encoder, run `bash scripts/run_price.sh`.

**1. Supervised fine-tuning**

```bash
python -m castreasoner.train sft \
  --model Qwen/Qwen3.5-9B \
  --train-file data/sft.parquet \
  --output-dir outputs/sft
```

**2. GRPO reinforcement learning**

Set `SFT_HF_MODEL` to the final SFT checkpoint exported in Hugging Face format, including its tokenizer. Supply a separate, nonempty validation file.

```bash
export SFT_HF_MODEL=/path/to/final/sft/hf_model

python -m castreasoner.train grpo \
  --model "$SFT_HF_MODEL" \
  --train-file data/rl.parquet \
  --val-file data/validation.parquet \
  --output-dir outputs/grpo
```

Add `--check-only` to validate inputs before training, or `--dry-run` to inspect the configuration. Stage settings are in [castreasoner/configs](castreasoner/configs), and the shared forecasting prompt is in [castreasoner/prompt.txt](castreasoner/prompt.txt).

## Offline SAD and teacher forcing

The paper's student-aware difficulty calculation is available as an independent utility in [castreasoner/sad.py](castreasoner/sad.py). `score_teacher_response` teacher-forces a complete teacher response through a frozen student, records token log-probabilities and vocabulary ranks for the visible `analysis`, `decision`, and `forecast_method` fields, and computes the field-balanced SAD score. `select_trajectory` applies the paper's MAE tolerance and same-strategy SAD tie-break. These utilities do not modify either training stage; call them from an offline trajectory-preparation script.
