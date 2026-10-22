import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ["pandas", "numpy", "scikit-learn", "vaderSentiment", "joblib", "streamlit"]
DEV_EXTRAS = ["matplotlib", "seaborn", "datasets"]


def pinned(packages: list) -> list:
    lines = []
    for package in packages:
        try:
            lines.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            print(f"STOP: {package} is not installed in this environment.")
            sys.exit(1)
    return lines


def main() -> None:
    runtime_lines = pinned(RUNTIME)
    dev_lines = ["-r requirements.txt"] + pinned(DEV_EXTRAS)

    (ROOT / "requirements.txt").write_text("\n".join(runtime_lines) + "\n", encoding="utf-8")
    (ROOT / "requirements-dev.txt").write_text("\n".join(dev_lines) + "\n", encoding="utf-8")

    print(f"Python used for training: {sys.version.split()[0]}")
    print("\nrequirements.txt:")
    print("\n".join("  " + line for line in runtime_lines))
    print("\nrequirements-dev.txt:")
    print("\n".join("  " + line for line in dev_lines))


if __name__ == "__main__":
    main()