# Conventions

## Development Environment

The project uses a devcontainer (`.devcontainer/`) on Ubuntu 24.04:
- Python 3.13 via `uv`
- Shell: zsh with Oh My Zsh
- Tools: ripgrep, fd-find (`fdfind`), fzf, ast-grep (`sg`), tmux, git-delta
- Shell aliases: `fd` → `fdfind`, `sg` → `ast-grep`

Container env (set in `devcontainer.json`):
- `PYTHONDONTWRITEBYTECODE=1` -- no `.pyc` files
- `UV_LINK_MODE=copy` -- uv uses copy instead of hardlinks

## License & Dependencies

- Project license: **MIT**
- NautilusTrader dependency: **LGPL-3.0-or-later** -- acceptable as unmodified library usage
  - Never modify NautilusTrader source code
  - Never vendor/copy NT source into this project
  - All custom code in separate `vibe_quant/` modules
- **Avoid AGPL** dependencies (license contamination risk)

## NautilusTrader Version Policy

- Pin to `major.minor`: `nautilus_trader>=1.222.0,<1.223.0`
- Accept patch releases for bugfixes
- Test before merging version bumps
- No source modifications (LGPL compliance)

## Database Conventions

### SQLite (State & Archive)

Every connection must set:
```sql
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
```

Use a `StateManager` class or connection factory that applies these automatically.

### Raw Data Archive

- All downloaded data goes into `data/archive/raw_data.db` **before** any processing
- Archive is immutable -- only append, never modify existing rows
- Catalog can be rebuilt from archive at any time via `python -m vibe_quant.data rebuild --from-archive`

## Technical Indicator Selection

1. **Prefer NautilusTrader built-in indicators** (Rust, high performance) for: RSI, EMA, SMA, MACD, Bollinger Bands, ATR, Stochastic
2. **Fall back to `pandas-ta-classic`** for indicators not available in NT
3. **Never use the original `pandas-ta`** package (compromised maintainership, supply chain risk)

## Code Organization

```
vibe_quant/
├── data/          # Data ingestion, archival, catalog management
├── dsl/           # Strategy DSL parser, validator, compiler
├── screening/     # Parameter sweep pipeline (NT screening mode)
├── validation/    # Full-fidelity backtesting (NT validation mode)
├── risk/          # Position sizing, risk management, circuit breakers
├── overfitting/   # DSR, Walk-Forward, Purged K-Fold filters
├── dashboard/     # Streamlit UI
└── alerts/        # Telegram bot integration
```

## Background Job Management

- Background backtests run as subprocesses tracked via `background_jobs` SQLite table
- Every subprocess writes heartbeat to SQLite every 30 seconds
- Dashboard cleans up stale jobs (no heartbeat for 120s) on startup
- Use PID tracking and `os.kill(pid, signal.SIGTERM)` for job cancellation

## Security

- API keys and secrets in environment variables only, never in code or config files
- Binance and Ethereal exchange credentials: env vars
- Telegram bot tokens: env vars

## Exchanges

- **Binance Futures** (USDT-M Perpetuals): Primary target, Phase 1+
- **Ethereal DEX**: Phase 7, requires custom EIP-712 signing, non-custodial model

## Git

- Git diffs use `delta` with line numbers
- No specific branching strategy required at this stage
