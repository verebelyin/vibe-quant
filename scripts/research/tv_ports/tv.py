"""tv.py -- render and emulate TradingView ports.

    render   --port DIR [--set k=v ...] [--timeframe TF]
        Print the port's DSL (YAML) with its parameter defaults, ``--set`` overrides applied,
        and the strategy timeframe replaced when ``--timeframe`` is given (the port refuses one outside
        its ``allowed_timeframes``).

    emulate  --port DIR --symbol S --start YYYY-MM-DD --end YYYY-MM-DD
             [--mode tv|engine] [--set k=v ...] [--timeframe TF]
        Replay the port's compiled strategy through the broker emulator and print the trade
        list and a summary. ``engine`` (default) reproduces the screening engine trade for
        trade; ``tv`` fills at the next bar's open the way TradingView does. The signals are
        the strategy's own (``vibe_quant.fast_tier.signals``), never recomputed here.

Run from the repository root with ``PYTHONPATH=$PWD .venv/bin/python scripts/research/tv_ports/tv.py ...``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load_dsl(port_path: str, sets: list[str], timeframe: str | None) -> dict[str, Any]:
    """The port's rendered DSL: defaults, then ``--set k=v`` (YAML-typed), then the timeframe.

    ``vibe_quant.discovery.port`` is resolved at call time: it lands with a sibling task, and
    ``render`` / ``emulate`` are the only code that needs it.
    """
    port_module = importlib.import_module("vibe_quant.discovery.port")
    port = port_module.load_port(port_path)
    values: dict[str, Any] = dict(port.defaults())
    for item in sets:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects k=v, got {item!r}")
        values[key] = yaml.safe_load(raw)
    # the port's own timeframe gate (``allowed_timeframes``) validates the override
    return dict(port.render(values, timeframe=timeframe))


def _fmt_ts(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def format_trades(result: Any) -> list[str]:
    """One line per position: entry, how it ended, net PnL (NT realized, commissions included)."""
    from tv_emulator import _cost_key

    lines: list[str] = []
    fills = list(result.fills)
    pos_iter = iter(result.positions)
    i = 0
    while i < len(fills):
        if fills[i].action != "entry":
            i += 1
            continue
        entry = fills[i]
        pos = next(pos_iter)
        # one position at a time: the first non-entry fill after an entry is the one that closes it
        later = [f for f in fills[i + 1 :] if f.action != "entry"]
        exit_fill = later[0] if later else None
        side = "LONG" if entry.side == "BUY" else "SHORT"
        ended = (
            f"{exit_fill.action} {exit_fill.price} @ {_fmt_ts(exit_fill.ts)}"
            if exit_fill is not None
            else "open at end"
        )
        cost = result.ledger.finalized.get(_cost_key(pos), 0.0)
        lines.append(
            f"{_fmt_ts(entry.ts)}  {side:<5} {entry.qty}  entry {entry.price}  -> {ended}"
            f"  pnl {float(pos.realized_pnl):.4f}  modeled costs {cost:.4f}"
        )
        i += 1
    return lines


def _emulate(args: argparse.Namespace) -> int:
    from tv_emulator import emulate

    from vibe_quant.screening.nt_runner import NTScreeningRunner

    dsl = _load_dsl(args.port, args.set, args.timeframe)
    runner = NTScreeningRunner(dsl, [args.symbol], args.start, args.end)
    result = emulate(runner, args.symbol, args.mode)
    for line in format_trades(result):
        print(line)
    equity = result.costed_equity()
    closed = sum(1 for p in result.positions if p.is_closed)
    print("--")
    print(
        f"mode={args.mode} symbol={args.symbol} {args.start}..{args.end} timeframe={dsl['timeframe']} "
        f"positions={len(result.positions)} closed={closed} fills={len(result.fills)}"
    )
    print(
        f"start={result.starting_balance:.2f} costed_equity_after_last_close="
        f"{(equity[-1] if equity else result.starting_balance):.4f} "
        f"margin_refusals={result.margin_refusals} reverse_dropped={result.reverse_dropped} "
        f"refresh_invalid={result.refresh_invalid}"
    )
    return 0


def _render(args: argparse.Namespace) -> int:
    dsl = _load_dsl(args.port, args.set, args.timeframe)
    sys.stdout.write(yaml.safe_dump(dsl, sort_keys=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("render", "emulate"):
        p = sub.add_parser(name)
        p.add_argument("--port", required=True, help="port folder (port.yaml)")
        p.add_argument("--set", action="append", default=[], metavar="k=v", help="parameter override")
        p.add_argument("--timeframe", help="replace the strategy timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
        if name == "emulate":
            p.add_argument("--symbol", required=True)
            p.add_argument("--start", required=True)
            p.add_argument("--end", required=True)
            p.add_argument("--mode", choices=("tv", "engine"), default="engine")
    args = parser.parse_args(argv)
    return _render(args) if args.command == "render" else _emulate(args)


if __name__ == "__main__":
    raise SystemExit(main())
