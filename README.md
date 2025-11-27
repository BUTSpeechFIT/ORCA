# 🐋 ORCA: Open-ended Response Correctness Assessment for Audio Question Answering

**Authors:** Šimon Sedláček¹\*, Sara Barahona²\*, Bolaji Yusuf¹\*, Laura Herrera-Alarcón²\*, Santosh Kesiraju¹\*, Cecilia Bolaños³, Alicia Lozano-Diez², Sathvik Udupa¹, Fernando López², Allison Ferner⁴, Ramani Duraiswami⁵, Jan Černocký¹

\* Equal contribution

¹Speech@FIT, Brno University of Technology, Czechia.
²Universidad Autónoma de Madrid, Spain. \
³University of Buenos Aires, Argentina.
⁴Tufts University, USA.
⁵University of Maryland, USA.

> **Status:** Paper under review for TACL (Transactions of the Association for Computational Linguistics)

ORCA is a framework for assessing the correctness of open-ended responses, particularly for audio question-answering tasks. The system uses language model representations and models the correctness of a response using Beta distribution thereby capturing both the mean and uncertainty (variance) of correctness score. The ORCA score strongly correlates with average human judgement and effectively captures the interpretive uncertainty.

## Coming Soon

- 🤗 **Pre-trained models** on HuggingFace
- 📊 **Training datasets** with 11,721 human annotations
- 🏆 **ORCA-based leaderboard** for audio QA model evaluation

## Features

- **Multiple scoring functions**: Bernoulli, Beta NLL, Beta Moment Matching (BMM), and MSE
- **Uncertainty estimation**: Models annotation variance using Beta distribution parameters (α, β)
- **LoRA fine-tuning**: Efficient parameter-efficient training with Low-Rank Adaptation


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
    --train_data path/to/train.json \
    --val_data path/to/val.json \
    --model google/gemma-3-1b-it \
    --score_type beta \
    --lora_rank 256 \
    --output_dir ./output \
    --log_dir ./logs
```

### Inference

```bash
orca-infer \
    --model_path ./output/best/model \
    --data path/to/test.json \
    --output_dir ./results
```

> **Note:** You can also use Python module syntax: `python -m orca_score.train` and `python -m orca_score.infer`

## Model Architecture

ORCA uses a pre-trained language model (e.g., Gemma, Llama, OLMo) with a linear scoring head that outputs log(α) and log(β) parameters for a Beta distribution. The Beta distribution captures:
- **Mean correctness score**: E[score] = α / (α + β)
- **Uncertainty/Variance**: Var[score] = (α·β) / ((α+β)²·(α+β+1))

## Key Arguments

### Model
- `--score_type`: Loss function (`beta`, `bernoulli`, `mse`, `bmm`)
- `--lora_rank`: LoRA rank for efficient fine-tuning
- `--init_type`: Linear layer initialization (`xavier`, `avg_emb`)
- `--layers_to_use`: Which transformer layers to use for scoring

### Training
- `--batch_size`: Per-device batch size
- `--accumulation_steps`: Gradient accumulation steps
- `--peak_lr`: Peak learning rate (default: 5e-5)
- `--max_steps`: Total training steps

## Repository Structure

```
orca_score/
├── model.py       # ORCA model implementation
├── train.py       # Training script
├── infer.py       # Inference script
├── data.py        # Dataset and data loading utilities
├── cli.py         # Command-line interface (orca-train, orca-infer)
└── utils.py       # Helper functions

tex/               # LaTeX source for TACL paper
```

## Citation

If you use ORCA in your research, please cite our TACL paper:

```bibtex
@article{sedlacek2025orca,
  title={ORCA: Open-ended Response Correctness Assessment for Audio Question Answering},
  author={Sedl\'{a}\v{c}ek, \v{S}imon and Barahona, Sara and Yusuf, Bolaji and Herrera-Alarc\'{o}n, Laura and Kesiraju, Santosh and Bola\~{n}os, Cecilia and Lozano-Diez, Alicia and Udupa, Sathvik and L\'{o}pez, Fernando and Ferner, Allison and Duraiswami, Ramani and \v{C}ernock\'{y}, Jan},
  journal={Transactions of the Association for Computational Linguistics},
  year={2025},
  note={Under review}
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
