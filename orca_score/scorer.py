"""Rate the correctness of a candidate answer, given the question and the reference answer.

    from orca_score.scorer import OrcaScorer, score

    # one pair (the model is loaded on the first call and kept for later calls)
    result = score(
        question="Does the alarm sound in front of the listener or behind him?",
        reference="behind",
        candidate="The alarm sounds behind the listener.",
        context="optional: a rationale and/or a transcript",
    )
    result["orca_score"]          # predicted correctness in [0, 1]
    result["orca_variance"]       # predicted variance of the human ratings (annotator disagreement)
    result["orca_score_clamped"]  # clamped score; None if no clamping threshold is set

    # many pairs, with a model of your choice
    scorer = OrcaScorer("gemma-4b")                                  # a published model
    scorer = OrcaScorer("/path/to/model_dir", var_threshold=0.05)    # a local directory
    results = scorer.score_many([{"question": ..., "reference": ..., "candidate": ...}, ...])

The model input is built by orca_score.data.format_model_input, the function the training data
loader uses, so the format is identical to training. This module holds the only inference loop of
the scoring tools (`predict`).
"""

import json
import os
from typing import Optional

import torch
import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from orca_score import data, hub
from orca_score import model as orca_model
from orca_score.clamp import clamp_orca_score
from orca_score.pairs import get_context

_DEFAULT_SCORERS: dict = {}


def load_model_and_tokenizer(model_path, tokenizer_path, device):
    """Load a trained ORCA model and its tokenizer.

    LoRA checkpoints are merged into the base language model; fully fine-tuned checkpoints are
    loaded directly. The model is loaded in bfloat16 on a GPU and in float32 on the CPU.

    Args:
        model_path: Path of the ORCA model directory (with config.yaml and lm/).
        tokenizer_path: Path of the tokenizer directory.
        device: torch.device to load the model on.

    Returns:
        (scoring_model, tokenizer)
    """
    model_path = os.path.normpath(str(model_path))
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    adapter_config = os.path.join(model_path, "lm", "adapter_config.json")
    if os.path.isfile(adapter_config):
        with open(adapter_config) as f:
            base_model_name = json.load(f)["base_model_name_or_path"]
        lm = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=dtype, device_map=device.type, low_cpu_mem_usage=True
        )
        lm = PeftModel.from_pretrained(lm, os.path.join(model_path, "lm"), low_cpu_mem_usage=True)
        lm = lm.merge_and_unload()  # type: ignore
    else:
        lm = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_path, "lm"),
            device_map=device.type,
            low_cpu_mem_usage=True,
            dtype=dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    scoring_model = orca_model.ORCA.load_from_directory(model_path, lm=lm, device=device)
    return scoring_model, tokenizer


def predict(model, collate_fn, items, device, batch_size=4, progress=False) -> list:
    """Run the model on prepared items. This is the inference loop.

    Items are processed in order of text length to reduce padding; the results are returned in
    the original order.

    Args:
        model: The ORCA scoring model (in eval mode).
        collate_fn: Callable that turns a list of items into a model batch with a .to() method.
        items: Items as built by OrcaScorer._make_item.
        device: torch.device the model is on.
        batch_size: Number of items per forward pass.
        progress: Show a progress bar.

    Returns:
        List of (score, variance) pairs, one per item.
    """
    order = sorted(range(len(items)), key=lambda i: len(items[i]["text"]))
    results: list = [None] * len(items)
    starts = range(0, len(order), batch_size)
    if progress:
        starts = tqdm.tqdm(starts, desc="Scoring", unit="batch")
    with torch.inference_mode():
        for start in starts:
            idx = order[start : start + batch_size]
            batch = collate_fn([items[i] for i in idx]).to(device)
            if device.type == "cpu":
                outputs = model(batch)
            else:
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    outputs = model(batch)
            scores = outputs["prob1"].float().cpu().tolist()
            variances = outputs["variance"].float().cpu().tolist()
            for i, s, v in zip(idx, scores, variances):
                results[i] = (s, v)
    return results


class OrcaScorer:
    """An ORCA model loaded for scoring.

    Args:
        model: Name of a published model (see hub.MODELS) or a local model directory.
        var_threshold: Variance threshold for clamping. Published models have one built in; for a
            local directory it must be given to get clamped scores. 0 disables clamping.
        tokenizer_path: Tokenizer directory, if it is not the tokenizer/ folder of the model.
        device: "cuda" or "cpu" (default: cuda if available).
        batch_size: Number of pairs scored at once.
        skip_context: Ignore the context of all pairs.
    """

    def __init__(
        self,
        model: str = "gemma-4b",
        var_threshold: Optional[float] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 4,
        skip_context: bool = False,
    ):
        spec = hub.resolve_model(model, tokenizer_path)
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        scoring_model, tokenizer = load_model_and_tokenizer(
            spec.model_path, spec.tokenizer_path, torch_device
        )
        threshold = var_threshold if var_threshold is not None else spec.var_threshold
        self._setup(
            scoring_model,
            tokenizer,
            spec.score_type,
            threshold,
            torch_device,
            batch_size,
            skip_context,
        )

    @classmethod
    def from_loaded(
        cls,
        model,
        tokenizer,
        score_type: str = "multinomial",
        var_threshold: Optional[float] = None,
        device: str = "cpu",
        batch_size: int = 4,
        skip_context: bool = False,
    ) -> "OrcaScorer":
        """Create a scorer from a model and tokenizer that are already loaded.

        Args:
            model: The ORCA scoring model.
            tokenizer: Its tokenizer.
            score_type: Score type of the model.
            var_threshold: Clamping threshold (None or 0: no clamping).
            device: Device the model is on.
            batch_size: Number of pairs scored at once.
            skip_context: Ignore the context of all pairs.

        Returns:
            The scorer.
        """
        scorer = cls.__new__(cls)
        scorer._setup(
            model,
            tokenizer,
            score_type,
            var_threshold,
            torch.device(device),
            batch_size,
            skip_context,
        )
        return scorer

    def _setup(self, model, tokenizer, score_type, threshold, device, batch_size, skip_context):
        """Store the loaded model and the settings (shared by both constructors)."""
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.score_type = score_type
        self.var_threshold = threshold if threshold else None
        self.device = device
        self.batch_size = batch_size
        self.skip_context = skip_context
        self._collate = data.CollateFn(tokenizer)

    def _make_item(self, pair: dict) -> dict:
        """Build one input item in the form the collate function expects.

        Args:
            pair: Dict with question, reference, candidate and optionally context.

        Returns:
            Item with the formatted text and placeholder ratings (not used for scoring).

        Raises:
            ValueError: If a required field is missing or not text.
        """
        for key in ("question", "reference", "candidate"):
            if not isinstance(pair.get(key), str):
                raise ValueError(f"'{key}' must be a text string")
        text = data.format_model_input(
            pair["question"],
            pair["reference"],
            pair["candidate"],
            context=get_context(pair),
            skip_rationale=self.skip_context,
        )
        dummy = 3.0 if self.score_type == "multinomial" else 0.5
        return {
            "text": text,
            "soft_labels": [dummy],
            "average_rating": dummy,
            "rating_variance": 0.0,
            "n_ratings": 1,
            "metadata": {},
            "prompt": "",
            "id": "",
        }

    def score_many(self, pairs: list, progress: bool = False) -> list:
        """Score several pairs.

        Args:
            pairs: List of dicts with `question`, `reference`, `candidate` and optionally
                `context` (a rationale and/or a transcript for the judge).
            progress: Show a progress bar.

        Returns:
            One dict per pair, in the same order, with orca_score, orca_variance and
            orca_score_clamped (None if no threshold is set).
        """
        items = [self._make_item(p) for p in pairs]
        predictions = predict(
            self.model, self._collate, items, self.device, self.batch_size, progress
        )
        results = []
        for s, v in predictions:
            clamped = (
                None if self.var_threshold is None else clamp_orca_score(s, v, self.var_threshold)
            )
            results.append({"orca_score": s, "orca_variance": v, "orca_score_clamped": clamped})
        return results

    def score(
        self, question: str, reference: str, candidate: str, context: Optional[str] = None
    ) -> dict:
        """Score one pair.

        Args:
            question: The question.
            reference: The correct answer.
            candidate: The answer to be judged.
            context: Optional rationale and/or transcript for the judge.

        Returns:
            Dict with orca_score, orca_variance and orca_score_clamped.
        """
        pair = {"question": question, "reference": reference, "candidate": candidate}
        if context is not None:
            pair["context"] = context
        return self.score_many([pair])[0]


def score(
    question: str,
    reference: str,
    candidate: str,
    context: Optional[str] = None,
    model: str = "gemma-4b",
    **scorer_kwargs,
) -> dict:
    """Score one pair, loading the model on the first call and keeping it for later calls.

    Args:
        question: The question.
        reference: The correct answer.
        candidate: The answer to be judged.
        context: Optional rationale and/or transcript for the judge.
        model: Model name or local directory (see OrcaScorer).
        **scorer_kwargs: Further OrcaScorer arguments (var_threshold, device, ...); they only
            matter for the call that loads the model.

    Returns:
        Dict with orca_score, orca_variance and orca_score_clamped.
    """
    if model not in _DEFAULT_SCORERS:
        _DEFAULT_SCORERS[model] = OrcaScorer(model, **scorer_kwargs)
    return _DEFAULT_SCORERS[model].score(question, reference, candidate, context)
