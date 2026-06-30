"""Download a pre-trained ORCA model and run inference on the stage-3 evaluation data.

Steps performed by this script:
  1. Download the selected ORCA model from BUT-FIT/<model> on Hugging Face Hub.
  2. Download the chosen stage-3 human-annotated JSONL from the dataset repo.
  3. Run orca-infer; results are written to the output directory.
  4. Apply variance-gated clamping via orca_score.clamp.

Use --stages to run only a subset of steps, e.g. --stages 3 4 to skip downloading.

Usage:
  python download_and_infer.py
  python download_and_infer.py --model gemma-4b --benchmark mmau-pro
  python download_and_infer.py --model olmo-1b --benchmark mmau-mmar --output_dir ./results
  python download_and_infer.py --stages 3 4 --model gemma-4b
  python download_and_infer.py --stages 4 --model olmo-1b --output_dir ./results
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_REPOS = {
    "gemma-4b": "BUT-FIT/orca-gemma-3-4b-it-multinomial",
    "llama-3b": "BUT-FIT/orca-llama-3.2-3b-it-multinomial",
    "olmo-1b": "BUT-FIT/orca-olmo-2-1b-multinomial",
}

# Variance thresholds calibrated on the dev set during paper experiments.
MODEL_VAR_THRESHOLDS = {
    "gemma-4b": 0.05,
    "llama-3b": 0.05,
    "olmo-1b": 0.005,
}

DATASET_REPO = "BUT-FIT/orca-audio-qa-annotations"

BENCHMARK_FILES = {
    "mmau-pro": "s3-mmau-pro-human-judge-ratings.jsonl",
    "mmau-mmar": "s3-mmau-mmar-human-judge-ratings.jsonl",
}


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download an ORCA model and run inference on stage-3 evaluation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REPOS.keys()),
        default="olmo-1b",
        help="Pre-trained model to download.",
    )
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARK_FILES.keys()),
        default="mmau-pro",
        help="Evaluation benchmark to run inference on.",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Local directory for the downloaded model. Defaults to orca-{model}.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Local directory for evaluation data.",
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory to write inference results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    parser.add_argument(
        "--var_threshold",
        type=float,
        default=None,
        help=(
            "Variance threshold for score clamping. "
            "Defaults to the value calibrated on the dev set during paper experiments "
            "(gemma-4b: 0.05, llama-3b: 0.05, olmo-1b: 0.005). "
            "Set to 0 to disable clamping."
        ),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        help=(
            "Which pipeline stages to run (default: all). "
            "1=download model, 2=download data, 3=inference, 4=clamping. "
            "Example: --stages 3 4 skips downloading and reruns inference+clamping."
        ),
    )
    return parser.parse_args()


def check_huggingface_hub():
    """Check that huggingface_hub is importable and return the module.

    Returns:
        module: The huggingface_hub module.
    """
    try:
        import huggingface_hub

        return huggingface_hub
    except ImportError:
        sys.exit(
            "ERROR: huggingface_hub is not installed.\n"
            "Install it with: pip install huggingface_hub"
        )


def download_model(repo_id, local_dir, hf_hub):
    """Download all files from a Hugging Face model repo.

    Args:
        repo_id: HF repo ID (e.g. 'BUT-FIT/orca-olmo-2-1b-multinomial').
        local_dir: Local directory path to download into.
        hf_hub: The huggingface_hub module.
    """
    local_dir = Path(local_dir)
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  {local_dir} already exists, skipping download.")
        return
    print(f"  Downloading {repo_id} -> {local_dir} ...")
    hf_hub.snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print("  Download complete.")


def download_eval_file(repo_id, filename, local_dir, hf_hub):
    """Download a single JSONL file from a Hugging Face dataset repo.

    Args:
        repo_id: HF dataset repo ID.
        filename: Filename to download from the repo root.
        local_dir: Local directory path to save the file into.
        hf_hub: The huggingface_hub module.

    Returns:
        Path: Path to the downloaded (or already-present) file.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / filename
    if dest.exists():
        print(f"  {filename} already exists, skipping.")
        return dest
    print(f"  Downloading {filename} ...")
    hf_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return dest


def find_orca_infer():
    """Locate the orca-infer executable in the current Python environment.

    Returns:
        str: Path to the orca-infer executable.
    """
    # Try the same bin directory as the running Python interpreter first.
    candidate = Path(sys.executable).parent / "orca-infer"
    if candidate.exists():
        return str(candidate)
    # Fall back to PATH lookup.
    found = shutil.which("orca-infer")
    if found:
        return found
    sys.exit(
        "ERROR: orca-infer not found.\n"
        "Install ORCA with: pip install -e . (from the repository root)"
    )


def run_inference(model_path, data_files, output_dir, batch_size, cpu):
    """Call orca-infer as a subprocess.

    Args:
        model_path: Path to the trained model directory.
        data_files: List of JSONL file paths to score.
        output_dir: Directory to write results to.
        batch_size: Inference batch size.
        cpu: If True, pass --cpu flag.
    """
    orca_infer = find_orca_infer()
    cmd = [
        orca_infer,
        "--model_path",
        str(model_path),
        "--output_dir",
        str(output_dir),
        "--batch_size",
        str(batch_size),
        "--data_jsonl",
    ] + [str(f) for f in data_files]
    if cpu:
        cmd.append("--cpu")
    print("  " + " \\\n    ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"ERROR: orca-infer exited with code {result.returncode}")


def run_clamping(result_dir, var_threshold):
    """Apply variance-gated score clamping to an orca-infer output directory.

    Calls orca_score.clamp in --apply_dir mode with the given threshold.
    Writes score_clamp.yaml into result_dir with clamped metrics.

    Args:
        result_dir: Directory containing final_result.jsonl (orca-infer output).
        var_threshold: Variance threshold; scores with variance <= threshold and
            extreme values (<0.125 or >0.875) are clamped to 0 or 1.
    """
    if var_threshold == 0.0:
        print("  var_threshold=0: clamping disabled, skipping.")
        return
    cmd = [
        sys.executable,
        "-m",
        "orca_score.clamp",
        "--apply_dir",
        str(result_dir),
        "--var_threshold",
        str(var_threshold),
    ]
    print("  " + " \\\n    ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"ERROR: orca_score.clamp exited with code {result.returncode}")


def main():
    """Download model and evaluation data, then run orca-infer and clamping."""
    args = parse_arguments()

    hf_hub = check_huggingface_hub()

    stages = set(args.stages)
    model_dir = args.model_dir or f"orca-{args.model}"
    model_path = Path(model_dir) / "model"
    filename = BENCHMARK_FILES[args.benchmark]
    var_threshold = (
        args.var_threshold if args.var_threshold is not None else MODEL_VAR_THRESHOLDS[args.model]
    )

    if 1 in stages:
        print(f"=== Step 1: Download model ({args.model}) ===")
        download_model(MODEL_REPOS[args.model], model_dir, hf_hub)

    if 2 in stages:
        print(f"\n=== Step 2: Download evaluation data ({args.benchmark}) ===")
        download_eval_file(DATASET_REPO, filename, args.data_dir, hf_hub)

    if 3 in stages:
        eval_path = Path(args.data_dir) / filename
        print("\n=== Step 3: Run inference ===")
        run_inference(model_path, [eval_path], args.output_dir, args.batch_size, args.cpu)

    # orca-infer writes results to output_dir/<stem>/final_result.jsonl
    result_dir = Path(args.output_dir) / Path(filename).stem

    if 4 in stages:
        print(f"\n=== Step 4: Apply clamping (var_threshold={var_threshold}) ===")
        run_clamping(result_dir, var_threshold)

    print(f"\nDone. Results are in: {args.output_dir}/")


if __name__ == "__main__":
    main()
