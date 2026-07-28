#!/usr/bin/env python
"""Notebook validation script.

Usage:
    uv run python scripts/validate_notebooks.py           # full validation
    uv run python scripts/validate_notebooks.py --fast    # fast mode (reduced steps)
    uv run python scripts/validate_notebooks.py --syntax-only  # syntax check only
"""
import subprocess, sys, os, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"


def check_notebook_syntax(nb_path: Path) -> bool:
    """Compile all code cells in a notebook to check for syntax errors."""
    name = nb_path.name
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    ok = True
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if not source.strip():
            continue
        # Strip Jupyter magic commands and shell commands (valid in notebook but not Python)
        clean_lines = []
        for line in source.split('\n'):
            stripped = line.lstrip()
            if stripped.startswith('%') or stripped.startswith('!') or stripped.startswith('?'):
                clean_lines.append('')  # Replace with blank line to preserve line numbers
            else:
                clean_lines.append(line)
        clean_source = '\n'.join(clean_lines)
        if not clean_source.strip():
            continue
        try:
            compile(clean_source, f"{name}:cell_{i}", 'exec')
        except SyntaxError as e:
            print(f"  SYNTAX ERROR in {name} cell {i}: {e}")
            ok = False
    return ok


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
    parser.add_argument("--syntax-only", action="store_true", help="Only check syntax, skip execution")
    args = parser.parse_args()

    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

    # Phase 1: Syntax check on ALL notebooks (always runs)
    print("=== Phase 1: Syntax Check (all notebooks) ===")
    syntax_ok = 0
    syntax_fail = 0
    for nb_path in notebooks:
        if check_notebook_syntax(nb_path):
            syntax_ok += 1
        else:
            syntax_fail += 1
    print(f"  Syntax: {syntax_ok} passed, {syntax_fail} failed out of {len(notebooks)}")

    if args.syntax_only:
        return 0 if syntax_fail == 0 else 1

    # Notebooks that require long training (skip in fast mode)
    long_notebooks = {"10_dqn.ipynb", "12_reinforce.ipynb", "14_actor_critic.ipynb",
                      "15_a2c.ipynb", "20_ppo_from_scratch.ipynb"}

    if args.fast:
        notebooks = [nb for nb in notebooks if nb.name not in long_notebooks]
        print(f"\n=== Phase 2: Fast Execution ({len(notebooks)} notebooks) ===")
    else:
        print(f"\n=== Phase 2: Full Execution ({len(notebooks)} notebooks) ===")

    if args.notebook:
        notebooks = [NOTEBOOK_DIR / args.notebook]

    passed, failed = 0, 0
    for nb_path in notebooks:
        if validate_notebook(nb_path, timeout=args.timeout):
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(notebooks)}")
    total_fail = syntax_fail + failed
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
