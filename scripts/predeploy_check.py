import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_MB = 50

REQUIRED = [
    "app.py", "orchestrator.py", "config.py", "evaluate.py", "requirements.txt", "README.md", ".gitignore",
    "agents/__init__.py", "agents/classifier.py", "agents/sentiment.py", "agents/knowledge.py",
    "agents/escalation.py", "agents/responder.py",
    "data/train.csv", "data/test.csv", "data/faq.csv", "data/eval_set.csv",
    "models/classifier.joblib",
    "results/baseline_comparison.csv", "results/baseline_vs_multiagent.png",
    "results/confusion_matrix_test.png",
]

failures = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    if not condition:
        failures.append(name)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(condition)


def tracked_files():
    try:
        outcome = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if outcome.returncode != 0:
        return None
    return [line for line in outcome.stdout.splitlines() if line.strip()]


def main() -> None:
    tracked = tracked_files()
    if not check("git is installed and this folder is a git repository", tracked is not None,
                 "install Git, then run: git init"):
        sys.exit(1)
    check("files are staged (run: git add .)", len(tracked) > 0)

    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    check("all required files exist", not missing, f"missing {missing}")
    not_tracked = [name for name in REQUIRED if (ROOT / name).exists() and name not in tracked]
    check("all required files are staged for git", not not_tracked,
          f"not staged: {not_tracked}. Check .gitignore, then git add .")

    unwanted = [path for path in tracked if path.startswith(("venv/", "data/raw/", "__pycache__/"))
                or path == "results/ticket_log.csv"]
    check("no venv, raw data or log file is tracked", not unwanted, str(unwanted[:5]))

    requirements = (ROOT / "requirements.txt")
    if check("requirements.txt exists", requirements.exists()):
        lines = [l.strip() for l in requirements.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        check("every line in requirements.txt is pinned with ==", bool(lines) and all("==" in l for l in lines),
              "run: python scripts\\make_requirements.py")

    sizes = sorted(((path, (ROOT / path).stat().st_size / 1_000_000) for path in tracked
                    if (ROOT / path).exists()), key=lambda item: item[1], reverse=True)
    too_big = [(p, round(s, 1)) for p, s in sizes if s > MAX_FILE_MB]
    check(f"no tracked file is larger than {MAX_FILE_MB} MB", not too_big, str(too_big))

    print(f"\nTracked files: {len(tracked)} | total size: {sum(s for _, s in sizes):.1f} MB")
    print("Largest tracked files:")
    for path, size in sizes[:5]:
        print(f"    {size:7.2f} MB  {path}")

    print()
    if failures:
        print(f"PREDEPLOY CHECK: FAIL ({len(failures)} problem(s)). Paste this output in the chat.")
        sys.exit(1)
    print("PREDEPLOY CHECK: PASS")


if __name__ == "__main__":
    main()