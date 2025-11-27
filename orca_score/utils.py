"""Authors: Bolaji Yusuf, Santosh Kesiraju"""

import json
import logging
import os

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


class CompositeWriter(SummaryWriter):
    """Writer that logs to TensorBoard, JSONL, and text logs.

    This class inherits from SummaryWriter and adds JSONL logging and text logging capabilities.

    Args:
        log_dir: Directory for TensorBoard logs (required).
        jsonl_path: Path to JSONL file for metrics logging (required).
        text_log_path: Path to text log file. If None, will be auto-generated from jsonl_path.
        resume: If True, appends to existing log. If False, rotates old log file.
        console_print_fn: Callable for console output. Defaults to print(). Pass accelerator.print for distributed training.
        **kwargs: Additional arguments passed to SummaryWriter.
    """

    def __init__(
        self, log_dir, jsonl_path, text_log_path=None, resume=False, console_print_fn=None, **kwargs
    ):
        self.jsonl_path = jsonl_path
        self.console_print_fn = console_print_fn or print

        # Setup text log path
        if text_log_path:
            self.text_log_path = text_log_path
        else:
            # Auto-generate text log path from jsonl_path
            log_dir_path = os.path.dirname(jsonl_path)
            self.text_log_path = os.path.join(log_dir_path, "training.log")

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.text_log_path), exist_ok=True)

        # Handle log rotation: rotate if file exists and NOT resuming
        # When resuming, we append to existing log. When restarting, we rotate old log.
        if os.path.exists(self.text_log_path) and not resume:
            self._rotate_log_file()

        # Setup text logger
        self.logger = logging.getLogger(f"orca_training_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()  # Clear any existing handlers

        # Use append mode if resuming, write mode if restarting
        mode = "a" if resume else "w"
        handler = logging.FileHandler(self.text_log_path, mode=mode)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False  # Prevent duplicate logs

        # Initialize TensorBoard
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
        """Log text message to file and optionally to console.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            to_console: If True, also prints to console using console_print_fn
        """
        # Always log to file
        if self.logger:
            log_func = getattr(self.logger, level.lower(), self.logger.info)
            log_func(message)

        # Optionally print to console
        if to_console:
            self.console_print_fn(message)

    def add_scalar(self, *args, **kwargs):
        """Add scalar to TensorBoard."""
        super().add_scalar(*args, **kwargs)

    def add_hparams(self, hparam_dict, metric_dict, **kwargs):
        """Log hyperparameters and metrics to both TensorBoard and JSONL.

        Args:
            hparam_dict: Dictionary of hyperparameters.
            metric_dict: Dictionary of metrics.
            **kwargs: Additional arguments passed to TensorBoard's add_hparams.
        """
        # Set run_name to empty string to prevent creating subdirectories
        kwargs.setdefault("run_name", ".")
        super().add_hparams(hparam_dict, metric_dict, **kwargs)

        # Always log to JSONL
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({**hparam_dict, **metric_dict}) + "\n")

    def flush(self):
        """Flush TensorBoard writer."""
        super().flush()

    def close(self):
        """Close TensorBoard writer and text logger."""
        super().close()
        if self.logger:
            for handler in self.logger.handlers:
                handler.close()
                self.logger.removeHandler(handler)


def save_checkpoint(
    scoring_model,
    output_dir,
    accelerator=None,
    optimizer=None,
    metrics=None,
    args=None,
    tokenizer=None,
):
    """
    Save the model and optimizer state to a checkpoint directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    if accelerator is None:
        scoring_model.save_to_directory(os.path.join(output_dir, "model"))
    else:
        accelerator.unwrap_model(scoring_model).save_to_directory(os.path.join(output_dir, "model"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    if args is not None:
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            yaml.dump(vars(args), f)
    if metrics is not None:
        with open(os.path.join(output_dir, "metrics.yaml"), "w") as f:
            yaml.dump(metrics, f)
