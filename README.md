# 🐋 ORCA: Open-ended Response Correctness Assessment for Audio Question Answering

**Authors:** Šimon Sedláček¹, Sara Barahona², Cecilia Bolaños³, Laura Herrera-Alarcón², Sathvik Udupa¹, Fernando López², Allison Ferner⁴, Alicia Lozano-Diez², Bolaji Yusuf¹, Santosh Kesiraju¹, Ramani Duraiswami⁵, Jan Černocký¹

¹Speech@FIT, Brno University of Technology, Czechia.
²Universidad Autónoma de Madrid, Spain. \
³University of Buenos Aires, Argentina.
⁴Tufts University, USA.
⁵University of Maryland, USA.

> **Status:** Paper accepted to TACL (Transactions of the Association for Computational Linguistics)
>
> **Paper:** The pre-MIT Press publication version is [on arXiv](https://arxiv.org/abs/2512.09066)

**Abstract:** Reliable assessment of the abilities of large audio language models (LALMs) is essential to advancing the state of the art. As benchmarks rapidly evolve to incorporate complex reasoning and subjective tasks, they increasingly necessitate open-ended responses from LALMs. We present Open-ended Response Correctness Assessment (ORCA) -- a reliable and lightweight model-based approach for answer correctness and disagreement modeling. We employ a three-stage annotation pipeline combining human judgment, structured feedback, and human-AI correction, yielding 9,663 annotations across 3,699 question-answer pairs from 15 LALMs on three audio understanding and reasoning benchmarks (achieving a Krippendorff's alpha of 0.82). Our experiments employing curriculum learning show that ORCA models achieve a Spearman correlation of 0.91 with average human correctness ratings on seen benchmarks and generalize to unseen benchmarks with a score of 0.85, outperforming several LLM judge baselines including Gemini 2.5 Flash. Furthermore, we demonstrate that ORCA's predicted variance correlates strongly with human disagreement, allowing it to effectively identify problematic benchmark items.

- **Training datasets** on [Hugging Face](https://huggingface.co/datasets/BUT-FIT/orca-audio-qa-annotations)
- **Pre-trained models** on Hugging Face:
  [orca-gemma-3-4b-it-multinomial](https://huggingface.co/BUT-FIT/orca-gemma-3-4b-it-multinomial) |
  [orca-llama-3.2-3b-it-multinomial](https://huggingface.co/BUT-FIT/orca-llama-3.2-3b-it-multinomial) |
  [orca-olmo-2-1b-multinomial](https://huggingface.co/BUT-FIT/orca-olmo-2-1b-multinomial)

## Installation

### Using uv (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment with Python 3.12+
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install ORCA
uv pip install -e .

# For development (includes black, ruff, isort, pytest, etc.)
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Python 3.12+ required
pip install -e .

# For development
pip install -e ".[dev]"
```

## Download and Inference

The quickest way to run ORCA is to download a pre-trained model and score your own data.

### 1. Install

```bash
pip install -e .
```

### 2. Download a model

```bash
# install the Hugging Face CLI if needed
pip install huggingface_hub

hf download BUT-FIT/orca-gemma-3-4b-it-multinomial --local-dir orca-gemma-4b
# alternatives:
# hf download BUT-FIT/orca-llama-3.2-3b-it-multinomial --local-dir orca-llama-3b
# hf download BUT-FIT/orca-olmo-2-1b-multinomial       --local-dir orca-olmo-1b
```

### 3. Prepare input data

Create a JSONL file with one item per line:

```json
{"id": "ex1", "question": "How many instruments are played?", "reference": "Three: piano, violin, and cello.", "candidate": "I hear piano and violin.", "rationale": "The candidate misses the cello.", "ratings": []}
```

Required fields: `question`, `reference`, `candidate`, `rationale`.
Add integer `ratings` (1–5) per annotator to enable automatic metric reporting.

### 4. Run inference

```bash
orca-infer \
    --model_path orca-gemma-4b/model \
    --data_jsonl your_data.jsonl \
    --output_dir results/
```

Output `results/your_data/final_result.jsonl` adds `rating_orca` (correctness score in [0,1]) and `variance_orca` to each input row. If `ratings` are provided, Spearman/Kendall/MAE metrics are printed.

### Using the provided evaluation splits

```bash
hf download BUT-FIT/orca-audio-qa-annotations \
    --type dataset --local-dir ./data \
    --include "stage3_human/seed_99/*"

orca-infer \
    --model_path orca-gemma-4b/model \
    --data_jsonl data/stage3_human/seed_99/test.jsonl \
    --output_dir results/
```

## Quick Start

### Training

```bash
orca-train \
    --train_data data/stage3_human/seed_99/train.jsonl \
    --val_data   data/stage3_human/seed_99/dev.jsonl \
    --model allenai/OLMo-2-0425-1B-Instruct \
    --score_type multinomial \
    --lora_rank 128 \
    --output_dir output/
```

### Inference

```bash
orca-infer \
    --model_path ./output/best/model \
    --data_jsonl data/stage3_human/seed_99/test.jsonl \
    --output_dir ./results/
```

## Model Architecture

ORCA uses a pre-trained language model (e.g., Gemma, Llama, OLMo) with a linear scoring head that outputs log(α) and log(β) parameters for a Beta distribution. The Beta distribution captures:
- **Mean correctness score**: E[score] = α / (α + β)
- **Uncertainty/Variance**: Var[score] = (α·β) / ((α+β)²·(α+β+1))

## Arguments

### Model
- `--model`: Pre-trained LM to use (e.g., `google/gemma-3-1b-it`, `meta-llama/Llama-3.2-1B`)
- `--score_type`: Loss function (`beta`, `multinomial`, `bernoulli`)
- `--lora_rank`: LoRA rank for efficient fine-tuning (omit for full fine-tuning)

### Training
- `--batch_size`: Per-device batch size (default: 8)
- `--accumulation_steps`: Gradient accumulation steps (default: 4)
- `--peak_lr`: Peak learning rate (default: 5e-5)
- `--max_steps`: Total training steps (default: 4000)
- `--early_stopping_patience`: Early stopping patience (default: 5)
- `--resume`: Resume from latest checkpoint in output_dir
- `--load_checkpoint`: Load model from specific checkpoint path

## Repository Structure

```
orca_score/
├── model.py       # ORCA model implementation
├── train.py       # Training script
├── infer.py       # Inference script
├── data.py        # Dataset and data loading utilities
├── cli.py         # Command-line interface (orca-train, orca-infer)
├── utils.py       # Helper functions
└── clamp.py       # Clamping only extreme ORCA outputs to hard zeros and ones.
```

## Citation

If you use ORCA or any of the curated datasets in your research, please cite our paper:

```bibtex
@article{sedlacek-etal-2026-orca,
  title={ORCA: Open-ended Response Correctness Assessment for Audio Question Answering},
  author={Sedl\'{a}\v{c}ek, \v{S}imon and Barahona, Sara and Herrera-Alarc\'{o}n, Laura and Kesiraju, Santosh and Bola\~{n}os, Cecilia and Lozano-Diez, Alicia and Udupa, Sathvik and L\'{o}pez, Fernando and Ferner, Allison and Yusuf, Bolaji and  Duraiswami, Ramani and \v{C}ernock\'{y}, Jan},
  howpublished={Accepted to Transactions of the Association for Computational Linguistics},
  year={2026},
  url={https://arxiv.org/abs/2512.09066}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact the corresponding authors:
- **Šimon Sedláček** - isedlacek@fit.vut.cz
- **Bolaji Yusuf** - iyusuf@fit.vut.cz
- **Santosh Kesiraju** - kesiraju@fit.vut.cz
