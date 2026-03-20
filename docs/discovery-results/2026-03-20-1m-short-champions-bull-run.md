# 1m Short Champion Validation on BTC Bull-Run Data

Date: 2026-03-20

## Goal

Test whether the current 1m SHORT champions still work in a clearly bullish BTC regime, or whether their edge is just an artifact of the recent bearish discovery window.

## Scenario

- Symbol: `BTCUSDT`
- Timeframe: `1m`
- Validation window: `2024-10-01` to `2024-12-31`
- Latency preset: `retail`
- Command:

```bash
.venv/bin/python -m vibe_quant validation batch \
  --strategy-ids 212,220 \
  --symbol BTCUSDT \
  --timeframe 1m \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --latency retail
```

## Why this counts as a bull run

The local raw 1m archive already covered this period, so no download was required. Over the test window BTC moved from `63,423.7` at the first 1m close to `92,750.2` at the last 1m close, a gain of roughly `+46.2%`. The period printed a local low of `58,900.0` and a high of `108,366.8` across `131,041` one-minute bars.

## Results

| Strategy | Bull run | Bearish reference | Verdict |
| --- | --- | --- | --- |
| `sid=212` (`genome_0ce860041f95`) | Run `755`: Sharpe `-4.14`, return `-17.06%`, max DD `20.98%`, `97` trades, WR `3.1%`, PF `0.49` | Run `710` on `2026-01-10..2026-03-10`: Sharpe `6.76`, return `+23.84%`, max DD `3.78%`, `50` trades, WR `16.0%`, PF `3.01` | Complete failure in an uptrend |
| `sid=220` (`genome_874e530c318a`) | Run `756`: Sharpe `-3.75`, return `-11.21%`, max DD `13.26%`, `91` trades, WR `82.4%`, PF `0.65` | Run `750` on `2026-01-17..2026-03-17`: Sharpe `3.91`, return `+7.55%`, max DD `4.16%`, `113` trades, WR `92.9%`, PF `1.71` | Still loses money despite high win rate |

## Interpretation

1. The recent 1m SHORT dominance is not a universal edge. It is strongly tied to the bearish 2026 discovery/validation window.
2. `sid=212` is the clearest evidence of period bias. It goes from an all-time 1m champion in the bearish window to a catastrophic loser in the bull window.
3. `sid=220` degrades differently: the win rate stays high, but the expectancy collapses. In a sustained uptrend, its many tiny short wins are overwhelmed by the larger stop-outs.
4. This means the current 1m program should not treat bearish-window short champions as robust until they survive at least one opposing regime.

## Conclusion

The answer to the bead's core question is clear: the 1m SHORT champions do **not** generalize to a strong BTC bull run. The observed 100% SHORT bias across the recent batches is a data-period artifact, not proof of a regime-agnostic short-side edge.

## Follow-up Issues

- `vibe-quant-7vty`: add a cross-regime validation gate before promoting 1m short champions
- `vibe-quant-y8wj`: run dedicated 1m bull-market discovery for long or bidirectional BTC strategies
