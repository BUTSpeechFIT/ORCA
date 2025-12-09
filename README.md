# 🐋 ORCA: Open-ended Response Correctness Assessment for Audio Question Answering

**Authors:** Šimon Sedláček¹\*, Sara Barahona²\*, Bolaji Yusuf¹\*, Cecilia Bolaños³\*, Laura Herrera-Alarcón²\*, Santosh Kesiraju¹\*, Alicia Lozano-Diez²\*, Sathvik Udupa¹, Fernando López², Allison Ferner⁴, Ramani Duraiswami⁵, Jan Černocký¹

\* Equal contribution

¹Speech@FIT, Brno University of Technology, Czechia.
²Universidad Autónoma de Madrid, Spain. \
³University of Buenos Aires, Argentina.
⁴Tufts University, USA.
⁵University of Maryland, USA.

> **Status:** Paper under review for TACL (Transactions of the Association for Computational Linguistics)
>
> 📄 **Paper:** [ORCA_paper.pdf](docs/ORCA_paper.pdf)

ORCA is a framework for assessing the correctness of open-ended responses, particularly for audio question-answering tasks. The system uses language model representations and models the correctness of a response using Beta distribution thereby capturing both the mean and uncertainty (variance) of correctness score. The ORCA score strongly correlates with average human judgement and effectively captures the interpretive uncertainty.

## Coming Soon

- 🤗 **Pre-trained models** on HuggingFace
- 📊 **Training datasets** with 11,721 human annotations
- 🏆 **ORCA-based leaderboard** for audio QA model evaluation

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

## Quick Start

### Training

```bash
orca-train \
    --train_data data/seed_108/train.json \
    --val_data data/seed_108/dev.json \
    --model allenai/OLMo-2-0425-1B-Instruct \
    --score_type beta \
    --lora_rank 256 \
    --output_dir output/
    --log_dir logs/
```

### Inference

```bash
orca-infer \
    --model_path ./output/best/model \
    --data data/seed_108/test.json \
    --output_dir ./results \
    --test_set_is_labeled
```

## Model Architecture

ORCA uses a pre-trained language model (e.g., Gemma, Llama, OLMo) with a linear scoring head that outputs log(α) and log(β) parameters for a Beta distribution. The Beta distribution captures:
- **Mean correctness score**: E[score] = α / (α + β)
- **Uncertainty/Variance**: Var[score] = (α·β) / ((α+β)²·(α+β+1))

## Arguments

### Model
- `--model`: Pre-trained LM to use (e.g., `google/gemma-3-1b-it`, `meta-llama/Llama-3.2-1B`)
- `--score_type`: Loss function (`beta`, `bernoulli`)
- `--lora_rank`: LoRA rank for efficient fine-tuning (omit for full fine-tuning)
- `--quantization_level`: Quantization (`none`, `4bit`, `8bit`)
- `--init_type`: Linear layer initialization for log(α), log(β) output:
  - `xavier_normal` (default): Xavier with small gain, starts near Beta(1,1) uniform
  - `kaiming_normal`: Kaiming scaled down, starts near Beta(1.1,1.1)
- `--use_cls_token`: Append learnable CLS token for scoring
- `--use_flash_attention`: Use Flash Attention 2 (requires `flash-attn` package)

### Training
- `--batch_size`: Per-device batch size (default: 1)
- `--accumulation_steps`: Gradient accumulation steps (default: 4)
- `--peak_lr`: Peak learning rate (default: 5e-5)
- `--max_steps`: Total training steps (default: 4000)
- `--val_steps`: Validation interval in steps (default: 200)
- `--save_steps`: Checkpoint save interval (default: 500)
- `--warmup_steps`: Learning rate warmup steps (default: 100)
- `--weight_decay`: Weight decay for optimizer (default: 0)
- `--lr_ratio_classifier`: LR ratio for scoring head vs LM (default: 1.0)
- `--early_stopping_patience`: Early stopping patience (default: 30)
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
└── utils.py       # Helper functions

tex/               # LaTeX source for paper
```

## Citation

If you use ORCA in your research, please cite our pre-print (under review):

```bibtex
@misc{sedlacek2025orca,
  title={ORCA: Open-ended Response Correctness Assessment for Audio Question Answering},
  author={Sedl\'{a}\v{c}ek, \v{S}imon and Barahona, Sara and Yusuf, Bolaji and Herrera-Alarc\'{o}n, Laura and Kesiraju, Santosh and Bola\~{n}os, Cecilia and Lozano-Diez, Alicia and Udupa, Sathvik and L\'{o}pez, Fernando and Ferner, Allison and Duraiswami, Ramani and \v{C}ernock\'{y}, Jan},
  howpublished={Manuscript under review for Transactions of the Association for Computational Linguistics},
  year={2025},
  url={https://github.com/BUTSpeechFIT/ORCA}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Speech@FIT, Brno University of Technology

## Contact

For questions or issues, please contact the corresponding authors:
- **Šimon Sedláček** - isedlacek@fit.vut.cz
- **Bolaji Yusuf** - iyusuf@fit.vut.cz
- **Santosh Kesiraju** - kesiraju@fit.vut.cz

**Primary Institution:** Speech@FIT, Brno University of Technology, Czechia

**Project Page:** https://github.com/BUTSpeechFIT/ORCA
