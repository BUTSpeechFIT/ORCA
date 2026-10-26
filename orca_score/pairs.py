"""Input format of the scoring tools: JSONL rows with a question, a reference and a candidate answer.

Each line is a JSON object with
  question, reference, candidate   required text: the question, the correct answer, the answer to judge
  context                          optional text for the judge: a rationale and/or a transcript
                                   (`rationale` is accepted as another name for it)
  id                               optional label, copied to the output (need not be unique)
  ratings                          optional human ratings (numbers from 1 to 5)
All other fields are kept and copied to the output.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REQUIRED_FIELDS = ("question", "reference", "candidate")
MAX_REPORTED_ERRORS = 10


@dataclass
class Pairs:
    """Rows read from an input file.

    Attributes:
        rows: The rows as dicts, each with an `id`.
        n_with_ratings: Number of rows that have at least one valid human rating.
    """

    rows: list
    n_with_ratings: int

    @property
    def labeled(self) -> bool:
        """True if every row has human ratings (so agreement metrics can be computed)."""
        return self.n_with_ratings == len(self.rows)


def clean_ratings(value) -> list:
    """Return the valid ratings (numbers from 1 to 5) of a ratings field.

    Args:
        value: The ratings field of one row (any JSON value).

    Returns:
        List of valid ratings; empty if there are none.
    """
    if not isinstance(value, list):
        return []
    return [
        r for r in value if isinstance(r, (int, float)) and not isinstance(r, bool) and 1 <= r <= 5
    ]


def get_context(row: dict) -> Optional[str]:
    """Return the context of a row: the `context` field, else the `rationale` field.

    Args:
        row: One input row.

    Returns:
        The context text, or None if the row has none.
    """
    context = row.get("context")
    return context if context is not None else row.get("rationale")


def check_row(row: dict) -> Optional[str]:
    """Check one row.

    Args:
        row: A parsed JSON object.

    Returns:
        A description of the problem, or None if the row is fine.
    """
    missing = [k for k in REQUIRED_FIELDS if not isinstance(row.get(k), str)]
    if missing:
        return f"missing or non-text field(s): {', '.join(missing)}"
    for key in ("context", "rationale"):
        if row.get(key) is not None and not isinstance(row[key], str):
            return f"'{key}' must be text"
    return None


def read_pairs(path: Path) -> Pairs:
    """Read and check an input JSONL file.

    All problems found are reported together, with line numbers.

    Args:
        path: Path of the JSONL file.

    Returns:
        The rows (each with an `id`) and the number of rows that have ratings.

    Raises:
        ValueError: If any line is invalid or the file has no rows.
    """
    rows, errors = [], []
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {number}: not valid JSON ({exc.msg})")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {number}: expected a JSON object")
                continue
            problem = check_row(row)
            if problem:
                errors.append(f"line {number}: {problem}")
                continue
            row["id"] = str(row["id"]) if row.get("id") is not None else f"row-{len(rows) + 1}"
            rows.append(row)
    if errors:
        shown = errors[:MAX_REPORTED_ERRORS]
        more = f"\n  ... and {len(errors) - len(shown)} more" if len(errors) > len(shown) else ""
        raise ValueError(f"{Path(path).name} has invalid rows:\n  " + "\n  ".join(shown) + more)
    if not rows:
        raise ValueError(f"{Path(path).name} has no rows")
    return Pairs(rows, sum(1 for r in rows if clean_ratings(r.get("ratings"))))


def write_jsonl(rows: list, path: Path) -> None:
    """Write rows as JSONL, creating the parent directory.

    Args:
        rows: List of dicts.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
