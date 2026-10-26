"""Command-line interface of ORCA-Score: the entry points `orca-train` and `orca-score`.

orca-score rates the correctness of candidate answers: each input row needs a question, the
reference (correct) answer and a candidate answer, and can have a context (a rationale and/or a
transcript). See README.md for the input and output format.

    orca-score --model gemma-4b --input my_data.jsonl --output scored.jsonl
    orca-score --model gemma-4b --input mmau-pro
    orca-score --model /path/to/model_dir --input my_data.jsonl --var_threshold 0.05
    orca-score --list

Authors: Bolaji Yusuf, Santosh Kesiraju
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from orca_score import hub, pairs
from orca_score.metrics import agreement_metrics


def orca_train():
    """Entry point for orca-train command."""
    try:
        from orca_score.train import main
    except ModuleNotFoundError as exc:
        sys.exit(f"ERROR: {exc}. Training needs extra packages: pip install 'orca-score[train]'")
    main()


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments of orca-score.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        prog="orca-score",
        description="Rate the correctness of a candidate answer, given the question and the "
        "reference (correct) answer, with a pre-trained ORCA model. Each input row needs a "
        "question, a reference answer and a candidate answer, and can have a context (a rationale "
        "and/or a transcript for the judge). The output is the predicted human correctness "
        "rating in [0, 1] and its variance (annotator disagreement) for each row.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Name of a published ORCA model (" + ", ".join(hub.MODELS) + ") or a local model "
        "directory. Required; there is no default model.",
    )
    parser.add_argument(
        "--input",
        help="JSONL file to score: a local path or one of the hosted files ("
        + ", ".join(hub.DATASETS)
        + ").",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file (default: <input name>.orca.jsonl in the current directory).",
    )
    parser.add_argument(
        "--var_threshold",
        type=float,
        default=None,
        help="Variance threshold for clamping. Published models have one built in; it is needed "
        "to get clamped scores with a local model directory. 0 disables clamping.",
    )
    parser.add_argument("--no_clamp", action="store_true", help="Do not write clamped scores.")
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Tokenizer directory (default: the tokenizer/ folder next to the model/ folder).",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--cpu", action="store_true", help="Run on the CPU (slow).")
    parser.add_argument(
        "--skip_context",
        action="store_true",
        help="Ignore the context field, even if the data has one.",
    )
    parser.add_argument(
        "--download",
        metavar="MODEL",
        help="Download a published model (and its base language model) for later use, then exit.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available models and files, then exit.",
    )
    args = parser.parse_args()
    if args.list or args.download:
        return args
    if not args.model:
        parser.error("--model is required (one of: " + ", ".join(hub.MODELS) + ", or a directory)")
    if not args.input:
        parser.error(
            "--input is required (a JSONL file or one of: " + ", ".join(hub.DATASETS) + ")"
        )
    return args


def list_resources() -> None:
    """Print the published models and the hosted input files."""
    print("ORCA models (--model):")
    for name, info in hub.MODELS.items():
        print(f"  {name:10s} {info['repo']}  (clamping threshold {info['var_threshold']})")
    print("  or the path of a local model directory")
    print("\nHosted input files (--input):")
    for name, filename in hub.DATASETS.items():
        print(f"  {name:10s} {hub.DATASET_REPO}/{filename}")
    print("  or the path of a local .jsonl file")


def download_models(name: str) -> None:
    """Download a published model and its base language model, then print its version.

    Args:
        name: Name of a published model.
    """
    try:
        spec = hub.resolve_model(name)
        base = json.loads((spec.model_path / "lm" / "adapter_config.json").read_text())
        from huggingface_hub import snapshot_download

        snapshot_download(base["base_model_name_or_path"])
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"Downloaded {spec.version}")


def print_metrics(metrics: dict) -> None:
    """Print the agreement with the human ratings as a small table.

    Args:
        metrics: Output of metrics.agreement_metrics.
    """
    print(f"\nAgreement with human ratings (n={metrics['n']}, ratings scaled to 0-1):")
    print(f"  {'':10s} {'Spearman':>9s} {'Kendall':>9s} {'MAE':>9s}")
    for name in ("raw", "clamped"):
        if name in metrics:
            m = metrics[name]
            print(f"  {name:10s} {m['rho']:9.4f} {m['tau']:9.4f} {m['mae']:9.4f}")


def orca_score_cmd() -> None:
    """Entry point for the orca-score command: score the input file and write the output."""
    args = parse_arguments()
    if args.list:
        list_resources()
        return
    if args.download:
        download_models(args.download)
        return

    try:
        input_path = hub.resolve_data(args.input)
        data = pairs.read_pairs(input_path)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"ERROR: {exc}")
    print(
        f"{len(data.rows)} rows read from {input_path.name}"
        + (" (with human ratings)" if data.labeled else ""),
        flush=True,
    )
    if 0 < data.n_with_ratings < len(data.rows):
        print(
            f"WARNING: only {data.n_with_ratings} of {len(data.rows)} rows have ratings; "
            "no metrics will be computed.",
            flush=True,
        )

    from orca_score.scorer import OrcaScorer  # imports torch: only needed from here on

    try:
        scorer = OrcaScorer(
            args.model,
            var_threshold=0 if args.no_clamp else args.var_threshold,
            tokenizer_path=args.tokenizer_path,
            device="cpu" if args.cpu else None,
            batch_size=args.batch_size,
            skip_context=args.skip_context,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"ERROR: {exc}")

    print(f"Model: {scorer.version}", flush=True)
    if scorer.var_threshold is not None:
        origin = "--var_threshold" if args.var_threshold else "published model"
        print(f"Clamping threshold {scorer.var_threshold} ({origin})", flush=True)
    elif not args.no_clamp and args.var_threshold != 0:
        print(
            "NOTE: no clamping threshold for this model directory, so only raw scores are "
            "written. Pass --var_threshold to also get clamped scores.",
            flush=True,
        )

    results = scorer.score_many(data.rows, progress=True)
    scored = []
    for row, result in zip(data.rows, results):
        out = {
            **row,
            "orca_score": result["orca_score"],
            "orca_variance": result["orca_variance"],
            "orca_model": result["orca_model"],
        }
        if scorer.var_threshold is not None:
            out["orca_score_clamped"] = result["orca_score_clamped"]
        scored.append(out)

    output = args.output or Path(f"{input_path.stem}.orca.jsonl")
    pairs.write_jsonl(scored, output)
    print(f"\nWrote {len(scored)} scored rows to {output}")

    if data.labeled:
        metrics = agreement_metrics(scored)
        print_metrics(metrics)
        metrics_path = output.with_suffix(".metrics.yaml")
        metrics_path.write_text(yaml.safe_dump(metrics))
        print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    orca_score_cmd()
