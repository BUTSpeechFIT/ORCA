"""Authors: Bolaji Yusuf, Santosh Kesiraju"""

import json
import logging
import os

import torch
import yaml
from rich.console import Console
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error
from torch.utils.tensorboard import SummaryWriter


class CompositeWriter(SummaryWriter):
    """Writer that logs to TensorBoard, JSONL, and text logs.

    On non-main processes all methods are no-ops, so it is safe to construct
    unconditionally and call without ``if accelerator.is_main_process`` guards.
    File I/O (log rotation, handler creation, TensorBoard init) only runs on
    the main process, eliminating multi-GPU race conditions.

    Args:
        log_dir: Directory for TensorBoard logs.
        jsonl_path: Path to JSONL file for metrics logging.
        is_main_process: Only this process does real I/O. Defaults to True.
        text_log_path: Path to text log file. If None, auto-generated alongside jsonl_path.
        resume: If True, appends to existing log. If False, rotates old log file.
        console_print_fn: Callable for console output. Defaults to print().
        **kwargs: Additional arguments passed to SummaryWriter.
    """

    def __init__(
        self,
        log_dir,
        jsonl_path,
        is_main_process=True,
        text_log_path=None,
        resume=False,
        console_print_fn=None,
        **kwargs,
    ):
        self._enabled = is_main_process
        self.console = Console()

        if not self._enabled:
            # Non-main process: store paths for reference but open no files.
            self.jsonl_path = jsonl_path
            log_dir_path = os.path.dirname(jsonl_path)
            self.text_log_path = text_log_path or os.path.join(log_dir_path, "training.log")
            self.logger = None
            # Initialise SummaryWriter to a temp dir so the object is valid,
            # but immediately close it so no files are created.
            import tempfile

            super().__init__(log_dir=tempfile.mkdtemp(), **kwargs)
            self.close()
            return

        self.jsonl_path = jsonl_path
        self.console_print_fn = console_print_fn or print

        # Setup text log path
        if text_log_path:
            self.text_log_path = text_log_path
        else:
            log_dir_path = os.path.dirname(jsonl_path)
            self.text_log_path = os.path.join(log_dir_path, "training.log")

        os.makedirs(os.path.dirname(self.text_log_path), exist_ok=True)

        # Rotate existing log file when starting fresh (not resuming).
        if os.path.exists(self.text_log_path) and not resume:
            self._rotate_log_file()

        # Setup text logger
        self.logger = logging.getLogger(f"orca_training_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        mode = "a" if resume else "w"
        handler = logging.FileHandler(self.text_log_path, mode=mode)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)
        self.logger.propagate = False

        os.makedirs(log_dir, exist_ok=True)
        super().__init__(log_dir=log_dir, **kwargs)

    def _rotate_log_file(self):
        """Rotate existing log file by appending a number."""
        base_path = self.text_log_path.rsplit(".", 1)[0]
        ext = self.text_log_path.rsplit(".", 1)[1] if "." in self.text_log_path else "log"

        # Find next available number
        num = 1
        while os.path.exists(f"{base_path}.{num}.{ext}"):
            num += 1

        # Rename existing log
        os.rename(self.text_log_path, f"{base_path}.{num}.{ext}")

    def log(self, message, level="INFO", to_console=False):
        """Log text message to file and optionally to console. No-op on non-main processes."""
        if not self._enabled:
            return
        if self.logger:
            log_func = getattr(self.logger, level.lower(), self.logger.info)
            log_func(message)
        if to_console:
            self.console.print(message)

    def add_scalar(self, *args, **kwargs):
        """Add scalar to TensorBoard. No-op on non-main processes."""
        if not self._enabled:
            return
        super().add_scalar(*args, **kwargs)

    def add_hparams(self, hparam_dict, metric_dict, **kwargs):
        """Log hyperparameters and metrics to TensorBoard and JSONL. No-op on non-main processes."""
        if not self._enabled:
            return
        kwargs.setdefault("run_name", ".")
        super().add_hparams(hparam_dict, metric_dict, **kwargs)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({**hparam_dict, **metric_dict}) + "\n")

    def flush(self):
        """Flush TensorBoard writer. No-op on non-main processes."""
        if not self._enabled:
            return
        super().flush()

    def close(self):
        """Close TensorBoard writer and text logger."""
        super().close()
        if self.logger:
            for handler in self.logger.handlers:
                handler.close()
                self.logger.removeHandler(handler)


def compute_metrics(
    annotations: list, predictions: list, variance_annotations: list, variance_predictions: list
) -> dict:
    """Compute correlation and error metrics between annotations and model predictions.

    Args:
        annotations: Ground-truth mean scores.
        predictions: Predicted mean scores (prob1).
        variance_annotations: Ground-truth score variances.
        variance_predictions: Predicted score variances.

    Returns:
        Dictionary with keys: mean_mae, mean_tau, mean_rho, variance_mae, variance_tau, variance_rho.
    """
    mean_tau = kendalltau(annotations, predictions)
    mean_rho = spearmanr(annotations, predictions)
    mean_mae = mean_absolute_error(annotations, predictions)
    variance_mae = mean_absolute_error(variance_annotations, variance_predictions)
    variance_tau = kendalltau(variance_annotations, variance_predictions)
    variance_rho = spearmanr(variance_annotations, variance_predictions)
    return {
        "mean_mae": float(mean_mae),
        "mean_tau": float(mean_tau.statistic),
        "mean_rho": float(mean_rho.statistic),
        "variance_mae": float(variance_mae),
        "variance_tau": float(variance_tau.statistic),
        "variance_rho": float(variance_rho.statistic),
    }


def save_checkpoint(
    scoring_model,
    output_dir,
    accelerator=None,
    optimizer=None,
    metrics=None,
    args=None,
    tokenizer=None,
):
    """Save model and optimizer state to a checkpoint directory.

    FSDP-compatible: all ranks must call this function concurrently.
    ``accelerator.get_state_dict()`` is a collective all-gather that requires
    every rank to participate; only rank 0 then writes files to disk.
    Single-GPU (accelerator=None) falls back to the original behaviour.
    """
    model_dir = os.path.join(output_dir, "model")

    if accelerator is None:
        # Single-GPU path — unchanged behaviour
        os.makedirs(output_dir, exist_ok=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        scoring_model.save_to_directory(model_dir)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    else:
        # FSDP path — gather full state dict across all ranks (collective)
        full_state_dict = accelerator.get_state_dict(scoring_model)
        # Only rank 0 writes files; other ranks have already contributed to the
        # all-gather and can return now.
        if not accelerator.is_main_process:
            return
        os.makedirs(output_dir, exist_ok=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        # Split into LM weights (saved via save_pretrained) and the linear head
        lm_state_dict = {
            k[len("lm.") :]: v for k, v in full_state_dict.items() if k.startswith("lm.")
        }
        non_lm_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith("lm.")}
        unwrapped = accelerator.unwrap_model(scoring_model)
        os.makedirs(os.path.join(model_dir, "lm"), exist_ok=True)
        with open(os.path.join(model_dir, "config.yaml"), "w") as f:
            yaml.dump(unwrapped.config, f)
        unwrapped.lm.save_pretrained(os.path.join(model_dir, "lm"), state_dict=lm_state_dict)
        torch.save(non_lm_state_dict, os.path.join(model_dir, "model_minus_lm.pt"))
        # Optimizer state is rank-local with FSDP; resume from FSDP runs is not
        # supported — skip to avoid saving a partial (rank-0-only) state dict.

    if args is not None:
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            yaml.dump(vars(args), f)
    if metrics is not None:
        with open(os.path.join(output_dir, "metrics.yaml"), "w") as f:
            yaml.dump(metrics, f)
