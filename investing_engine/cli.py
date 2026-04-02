from __future__ import annotations

import argparse
import json

from investing_engine.pipeline.engine import InvestingEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the institutional AI investing engine prototype.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    engine = InvestingEngine()
    result = engine.run_once().to_json_dict()
    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
