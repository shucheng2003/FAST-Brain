"""Check Python syntax and the files required by the bundled datasets."""

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    sources += list((ROOT / "fast_brain").glob("*.py")) + [ROOT / "sitecustomize.py"]
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for dataset in ("rnn", "sddes"):
        config = yaml.safe_load((ROOT / "configs" / f"{dataset}.yaml").read_text(encoding="utf-8"))
        data = config["data"]
        inputs = [data["signals"], *data["matrices"]]
        if "real_ec" in data:
            inputs.append(data["real_ec"])
        for item in inputs:
            path = ROOT / item["path"]
            if not path.is_file():
                raise SystemExit(f"Missing {dataset} input: {path}")
    print("Python syntax and bundled dataset paths checked successfully.")


if __name__ == "__main__":
    main()
