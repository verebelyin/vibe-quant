"""``python -m vibe_quant.bankroll report --report-id R --job-id J --db PATH``.

Runs one admitted bankroll report as its background job (see
:mod:`vibe_quant.bankroll.report`). The catalog and archive default to the project's
``data/catalog`` and ``data/archive/raw_data.db`` (cwd-relative, as validation's); every
engine reads only the report's own snapshot of them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from vibe_quant.bankroll.report import default_scratch_base, run_report
    from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

    parser = argparse.ArgumentParser(prog="python -m vibe_quant.bankroll")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="run one admitted bankroll report")
    rep.add_argument("--report-id", type=int, required=True)
    rep.add_argument("--job-id", type=int, required=True, help="background_jobs.run_id")
    rep.add_argument("--db", type=Path, required=True, help="state database path")
    rep.add_argument("--catalog", type=Path, default=Path(DEFAULT_CATALOG_PATH))
    rep.add_argument("--archive", type=Path, default=Path(DEFAULT_ARCHIVE_PATH))
    rep.add_argument("--scratch", type=Path, default=None, help="scratch base (default: tmp)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run_report(
        args.report_id,
        args.job_id,
        args.db,
        catalog=args.catalog,
        archive=args.archive,
        scratch=args.scratch or default_scratch_base(),
    )


if __name__ == "__main__":
    sys.exit(main())
