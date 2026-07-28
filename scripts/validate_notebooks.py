#!/usr/bin/env python
"""Notebook validation script.

Usage:
    uv run python scripts/validate_notebooks.py           # full validation
    uv run python scripts/validate_notebooks.py --fast    # fast mode (reduced steps)
"""
import subprocess, sys, os, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"


def validate_notebook(nb_path: Path, timeout: int = 600) -> bool:
    """Execute a single notebook and return True if successful."""
    name = nb_path.name
    print(f"  Validating {name}...", end=" ", flush=True)
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute",
                "--output", f"/tmp/nbval_{name}",
                "--output-dir", "/tmp",
                "--ExecutePreprocessor.timeout", str(timeout),
                str(nb_path),
            ],
            capture_output=True, text=True, timeout=timeout + 60,
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            print("PASS")
            return True
        else:
            print(f"FAIL")
            # Print last 20 lines of error
            lines = result.stderr.split("\n")[-20:]
            for line in lines:
                if line.strip():
                    print(f"    {line.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({timeout}s)")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Fast mode (skip long notebooks)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per notebook (seconds)")
    parser.add_argument("--notebook", type=str, help="Validate a specific notebook")
    args = parser.parse_args()

    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

    # Notebooks that require long training (skip in fast mode)
    long_notebooks = {"10_dqn.ipynb", "12_reinforce.ipynb", "14_actor_critic.ipynb",
                      "15_a2c.ipynb", "20_ppo_from_scratch.ipynb"}

    if args.fast:
        notebooks = [nb for nb in notebooks if nb.name not in long_notebooks]
        print(f"Fast mode: validating {len(notebooks)} notebooks (skipping long training notebooks)")
    else:
        print(f"Validating {len(notebooks)} notebooks")

    if args.notebook:
        notebooks = [NOTEBOOK_DIR / args.notebook]

    passed, failed = 0, 0
    for nb_path in notebooks:
        if validate_notebook(nb_path, timeout=args.timeout):
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(notebooks)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
