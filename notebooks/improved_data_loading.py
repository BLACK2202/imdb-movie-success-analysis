"""Compatibility wrapper for the shared dataset preparation service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import settings
from app.services.data_loader import prepare_dataset



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=settings.default_chunksize)
    parser.add_argument("--force", action="store_true", help="Rebuild the prepared dataset even if it exists.")
    args = parser.parse_args()

    df = prepare_dataset(chunksize=args.chunksize, force=args.force)
    print(f"Prepared dataset rows: {len(df):,}")
    print(f"Prepared data directory: {settings.processed_data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
