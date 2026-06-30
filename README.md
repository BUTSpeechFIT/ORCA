# ORCA: Open-ended Response Correctness Assessment for Audio Question Answering

**Authors:** Šimon Sedláček¹, Sara Barahona², Cecilia Bolaños³, Laura Herrera-Alarcón², Sathvik Udupa¹, Fernando López², Allison Ferner⁴, Alicia Lozano-Diez², Bolaji Yusuf¹, Santosh Kesiraju¹, Ramani Duraiswami⁵, Jan Černocký¹

¹Speech@FIT, Brno University of Technology, Czechia &nbsp;·&nbsp;
²Universidad Autónoma de Madrid, Spain &nbsp;·&nbsp;
³University of Buenos Aires, Argentina &nbsp;·&nbsp;
⁴Tufts University, USA &nbsp;·&nbsp;
⁵University of Maryland, USA

**Paper:** [arXiv:2512.09066](https://arxiv.org/abs/2512.09066) — accepted to *TACL 2026*  
**Dataset:** [BUT-FIT/orca-audio-qa-annotations](https://huggingface.co/datasets/BUT-FIT/orca-audio-qa-annotations)  
**Models:** [orca-gemma-3-4b-it-multinomial](https://huggingface.co/BUT-FIT/orca-gemma-3-4b-it-multinomial) · [orca-llama-3.2-3b-it-multinomial](https://huggingface.co/BUT-FIT/orca-llama-3.2-3b-it-multinomial) · [orca-olmo-2-1b-multinomial](https://huggingface.co/BUT-FIT/orca-olmo-2-1b-multinomial)

---

**Abstract:** Reliable assessment of the abilities of large audio language models (LALMs) is essential to advancing the state of the art. As benchmarks rapidly evolve to incorporate complex reasoning and subjective tasks, they increasingly necessitate open-ended responses from LALMs. We present Open-ended Response Correctness Assessment (ORCA) — a reliable and lightweight model-based approach for answer correctness and disagreement modeling. We employ a three-stage annotation pipeline combining human judgment, structured feedback, and human-AI correction, yielding 9,663 annotations across 3,699 question-answer pairs from 15 LALMs on three audio understanding and reasoning benchmarks (achieving a Krippendorff's alpha of 0.82). Our experiments employing curriculum learning show that ORCA models achieve a Spearman correlation of 0.91 with average human correctness ratings on seen benchmarks and generalize to unseen benchmarks with a score of 0.85, outperforming several LLM judge baselines including Gemini 2.5 Flash. Furthermore, we demonstrate that ORCA's predicted variance correlates strongly with human disagreement, allowing it to effectively identify problematic benchmark items.

---

## Installation

```bash
# Python 3.12+ required
pip install -e .
```

Using [uv](https://docs.astral.sh/uv/) (recommended for development):

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Inference with pre-trained models

The `download_and_infer.py` script handles everything — model download, data download, inference, and score clamping:

```bash
python download_and_infer.py --model gemma-4b --benchmark mmau-pro
```

Available options:
- `--model`: `gemma-4b` (default: `olmo-1b`), `llama-3b`, `olmo-1b`
- `--benchmark`: `mmau-pro` (default), `mmau-mmar`
- `--stages 1 2 3 4`: run all steps (default) or a subset — e.g. `--stages 3 4` to skip downloading
- `--var_threshold`: override the clamping threshold (default: per-model calibrated value)

Results are written to `results/<model>/<benchmark>/final_result.jsonl` (`rating_orca`, `variance_orca` per item) and `score_clamp.yaml` (Spearman, Kendall, MAE).

### Manual steps

```bash
# 1. Download a model
pip install huggingface_hub
hf download BUT-FIT/orca-gemma-3-4b-it-multinomial --local-dir orca-gemma-4b

# 2. Prepare input JSONL (one item per line)
# Required fields: question, reference, candidate, rationale
# Optional: ratings (list of int 1–5) to get Spearman/Kendall/MAE metrics

# 3. Run inference
orca-infer --model_path orca-gemma-4b/model --data_jsonl your_data.jsonl --output_dir results/

# 4. Apply clamping
python -m orca_score.clamp --apply_dir results/your_data --var_threshold 0.05
```

## Training

```bash
orca-train \
    --train_data /path/to/train.jsonl \
    --val_data   /path/to/dev.jsonl \
    --model allenai/OLMo-2-0425-1B-Instruct \
    --score_type multinomial \
    --lora_rank 128 \
    --output_dir output/
```

Training data is available at [BUT-FIT/orca-audio-qa-annotations](https://huggingface.co/datasets/BUT-FIT/orca-audio-qa-annotations). To reproduce the full three-stage curriculum, train sequentially on `stage1_pretrain` → `stage2_benchmark` → `stage3_mmau_mmar`, passing `--load_checkpoint` between stages.

Run `orca-train --help` and `orca-infer --help` for the full argument reference.

## Model architecture

ORCA fine-tunes a pre-trained LM (Gemma, Llama, OLMo, …) with a LoRA adapter and a small linear scoring head that models the distribution over a 5-point Likert scale, from which a continuous correctness score in [0, 1] and a variance estimate are derived. Score clamping (`orca_score/clamp.py`) optionally hard-clamps extreme, low-variance predictions to 0 or 1.

## Repository structure

```
orca_score/
├── model.py    # ORCA model
├── train.py    # Training loop (orca-train)
├── infer.py    # Inference (orca-infer)
├── data.py     # Data loading
├── clamp.py    # Score clamping
├── cli.py      # Entry points
└── utils.py    # Helpers
download_and_infer.py   # End-to-end convenience script
```

## Citation

```bibtex
@article{sedlacek-etal-2026-orca,
  title={ORCA: Open-ended Response Correctness Assessment for Audio Question Answering},
  author={Sedl\'{a}\v{c}ek, \v{S}imon and Barahona, Sara and Herrera-Alarc\'{o}n, Laura and Kesiraju, Santosh and Bola\~{n}os, Cecilia and Lozano-Diez, Alicia and Udupa, Sathvik and L\'{o}pez, Fernando and Ferner, Allison and Yusuf, Bolaji and Duraiswami, Ramani and \v{C}ernock\'{y}, Jan},
  howpublished={Accepted to Transactions of the Association for Computational Linguistics},
  year={2026},
  url={https://arxiv.org/abs/2512.09066}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Contact

- **Šimon Sedláček** — isedlacek@fit.vut.cz
- **Bolaji Yusuf** — iyusuf@fit.vut.cz
- **Santosh Kesiraju** — kesiraju@fit.vut.cz
