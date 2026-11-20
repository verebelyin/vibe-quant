"""Bankroll sizing report for a completed validation run (bead vibe-quant-2gyk1.3).

Pure pieces, importable without the engine, validation or the state DB:

- :mod:`.kelly`: continuous Kelly ``mu/sigma^2``, half-Kelly and vol-target numbers on the
  run's own daily mark-to-market returns.
- :mod:`.stress`: the size grid at the parent's margin settings (risk held at the 50%
  sizing-budget limit) and each row's liquidation facts.
- :mod:`.recommend`: row eligibility and a recommendation that is always a tested row.
- :mod:`.snapshot`: the report-local market snapshot every engine of one report reads.

This package module imports none of them, so importing one stays light.
"""
