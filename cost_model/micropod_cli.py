"""Run from LLMCompassPlus: python -m cost_model.micropod_cli --config ... --output ..."""

import argparse
import json
from pathlib import Path

from .micropod import estimate_micropod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = estimate_micropod(config, config_dir=args.config.resolve().parent)
        output = (
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
