import sys


def check_library(import_name: str, display_name: str = None) -> bool:
    display_name = display_name or import_name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "version not exposed")
        print(f"[OK]   {display_name:<18} {version}")
        return True
    except ImportError as error:
        print(f"[FAIL] {display_name:<18} NOT INSTALLED ({error})")
        return False


def main() -> None:
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python path   : {sys.executable}")
    print("-" * 50)

    libraries = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("vaderSentiment", "vaderSentiment"),
        ("joblib", "joblib"),
        ("streamlit", "streamlit"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("datasets", "datasets"),
    ]

    results = [check_library(name, label) for name, label in libraries]

    print("-" * 50)
    if all(results):
        print("SUCCESS: all libraries imported correctly.")
    else:
        print("PROBLEM: some libraries failed. See [FAIL] lines above.")


if __name__ == "__main__":
    main()