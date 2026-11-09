"""Write the JSON schema of ExtractionPage to a versioned file, used by runner_vision.

Example:
    python -m src.schemas.export_schema
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.schemas.extraction import ExtractionPage

DEFAULT_OUT = Path("conf/schemas/extraction_page.json")


def schema_text() -> str:
    """The schema as written to disk (stable key order, French data kept as is)."""
    return json.dumps(ExtractionPage.model_json_schema(), indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Export the ExtractionPage JSON schema")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output file (default {DEFAULT_OUT})")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(schema_text(), encoding="utf-8")
    print(f"schema written to {args.out}")


if __name__ == "__main__":
    main()
