# Draft upstream issue for nautechsystems/nautilus_trader

Status: ready to post (vibe-quant-k448b). Verified against 1.230.0.

---

**Title:** `BacktestDataConfig` with string `data_cls` silently disables bar-type narrowing — loads the entire catalog

**Body:**

## Bug description

`BacktestDataConfig.query` decides identifier narrowing with an identity
comparison against the raw `data_cls` field:

```python
# nautilus_trader/backtest/config.py
@property
def query(self) -> dict[str, Any]:
    identifiers = []
    if self.data_cls is Bar:          # <-- fails when data_cls is a string
        if self.bar_types:
            identifiers = [str(bar_type) for bar_type in self.bar_types]
        elif self.instrument_id and self.bar_spec:
            ...
```

`data_cls` is documented to accept an import path string (and the docs'
own examples use `data_cls="nautilus_trader.model.data:Bar"` /
`data_cls=Bar.fully_qualified_name()`). With the string form,
`self.data_cls is Bar` is `False`, so:

- `bar_types=[...]` is **silently ignored** → `identifiers = []` → the
  one-shot loader (`BacktestNode._run_oneshot` → `load_data_config` →
  `catalog.query(**config.query)`) loads **every Bar dataset in the
  catalog**, all instruments, all timeframes.
- `instrument_id="X"` + `bar_spec="4-HOUR-LAST"` falls through to the
  instrument-only fallback → loads **every bar timeframe** for that
  instrument.

The streaming path (`chunk_size` set) resolves `config.data_type`
(which imports the string) and is unaffected — only the default
one-shot path is broken.

## Impact

For a catalog holding 1m/5m/15m/1h/4h bars for 3 instruments, a
single-instrument 4h backtest (4,836 bars needed) iterated 1,494,324
events (~300x). Runtime and results both affected: the venue processes
the stray finer-granularity bars, so fills/stops trigger intrabar
rather than on the requested bars.

## Reproduction

```python
from nautilus_trader.config import BacktestDataConfig
from nautilus_trader.model.data import Bar

kwargs = dict(
    catalog_path="catalog",
    bar_types=["BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL"],
    start_time="2024-01-01",
    end_time="2026-03-17",
)

print(BacktestDataConfig(data_cls="nautilus_trader.model.data:Bar", **kwargs).query["identifiers"])
# []            <-- bug: bar_types dropped

print(BacktestDataConfig(data_cls=Bar, **kwargs).query["identifiers"])
# ['BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL']
```

## Suggested fix

Compare against the resolved class in `query` (and anywhere else that
checks `self.data_cls is Bar`):

```python
if self.data_type is Bar:
```

## Version

- nautilus_trader 1.230.0 (also reproduced on 1.222.0)
- Python 3.13, macOS arm64
