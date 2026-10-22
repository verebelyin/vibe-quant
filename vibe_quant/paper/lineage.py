"""Checkpoint lineage: which saved session a start carries.

The restore (:meth:`~vibe_quant.paper.node.PaperTradingNode._restore_from_checkpoint`)
and the LIVE consent gate ask the same questions of the same rows, so they live in
one place, over the caller's :class:`~vibe_quant.paper.persistence.StatePersistence`
connection. Behaviour-preserving move (vibe-quant-rabmo.1): every SQL text, ORDER
BY, LIMIT and parameter order is the one the store used before the move, and the
SQL literals keep their original spelling so a diff against the old methods is empty.

Callers import the module (``from vibe_quant.paper import lineage``) and call
``lineage.has_prior_state(...)``: a monkeypatch on a module attribute then reaches
every caller at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence


#: ae6ah: the trader's rows in one venue + env: any key of that env, a legacy row (no /
#: null account), an unreadable row, or a non-string account (the last two fail closed
#: when parsed or checked). One predicate for the consent gate and the restore, so
#: "the gate asks for consent" and "nothing carries" cannot drift apart. Parameters:
#: ``(trader_id, account_env(account) + ":", account_env(account) + ":")``.
_ENV_ROW_WHERE = """trader_id = ?
  AND CASE WHEN json_valid(node_status)
           THEN CASE json_type(node_status, '$.account')
                    WHEN 'text' THEN substr(json_extract(node_status, '$.account'), 1, length(?)) = ?
                    ELSE 1 END
           ELSE 1 END"""


def account_env(account: str) -> str:
    """``binance:testnet:<hash>`` -> ``binance:testnet`` (qa42q: same venue + env)."""
    return account.rpartition(":")[0]


def is_key_rotation(newest_account: object, account: str) -> bool:
    """Is the trader's newest checkpoint account a KEY ROTATION into ``account``?

    True when ``newest_account`` is another account of the same venue + env (another
    key). The one rule the LIVE consent gate and the restore share (vibe-quant-neuvu.2),
    so "the gate asks for consent" and "nothing carries" cannot drift apart.

    Raises:
        TypeError: ``newest_account`` is neither None nor a string (a corrupt row);
            the restore fails closed on it, the gate treats it as "no rotation".
    """
    if newest_account is None:
        return False
    if not isinstance(newest_account, str):
        raise TypeError(f"checkpoint account must be a string, not {type(newest_account).__name__}")
    return newest_account != account and account_env(newest_account) == account_env(account)


def latest_own(
    store: StatePersistence, trader_id: str, account: str
) -> StateCheckpoint | None:
    """Newest checkpoint of ``trader_id`` that belongs to ``account``.

    A row belongs to ``account`` when its ``account`` key equals it or is missing
    (legacy); rows of another account (testnet vs live, another key) are skipped.
    A row whose ``node_status`` is not valid JSON counts (it fails closed when
    parsed). Same predicate as the own-id threshold of :func:`newer_other`.
    """
    cursor = store.conn.execute(
        """SELECT * FROM paper_trading_checkpoints
               WHERE trader_id = ?
                 AND CASE WHEN json_valid(node_status)
                          THEN COALESCE(json_extract(node_status, '$.account'), ?)
                          ELSE ? END = ?
               ORDER BY id DESC
               LIMIT 1""",
        (trader_id, account, account, account),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return store.row_to_checkpoint(row)


def latest_in_env(
    store: StatePersistence, trader_id: str, account: str
) -> StateCheckpoint | None:
    """Newest checkpoint of ``trader_id`` in ``account``'s venue + env (any key).

    Ae6ah (D1=A): rotating back A -> B -> A restores the newest row of the env,
    so an older key's own row is never hidden by a row of the other env. Parsed
    strictly (an unreadable row raises, the restore fails closed).
    """
    prefix = account_env(account) + ":"
    cursor = store.conn.execute(
        f"""SELECT * FROM paper_trading_checkpoints
               WHERE {_ENV_ROW_WHERE}
               ORDER BY id DESC
               LIMIT 1""",
        (trader_id, prefix, prefix),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return store.row_to_checkpoint(row)


def newer_other(
    store: StatePersistence, account: str, trader_id: str
) -> StateCheckpoint | None:
    """Newest checkpoint of ANOTHER trader on ``account`` newer than ``trader_id``'s own.

    "Newer" is a higher row id than ``trader_id``'s newest row ON THIS
    ``account`` (0 when it has none; its rows without an ``account`` key count as
    this account, rows of another account do not). Rows whose ``node_status`` is not valid JSON are skipped (their
    account cannot be known; ``json_valid`` makes that independent of scan
    order). The CHOSEN row is parsed strictly: a parse failure raises, so the
    caller fails closed.

    Returns:
        The checkpoint, or None if no other trader on the account is newer.
    """
    cursor = store.conn.execute(
        """SELECT * FROM paper_trading_checkpoints
               WHERE trader_id != ?
                 AND json_valid(node_status)
                 AND json_extract(node_status, '$.account') = ?
                 AND id > (SELECT COALESCE(MAX(id), 0) FROM paper_trading_checkpoints
                           WHERE trader_id = ?
                             AND CASE WHEN json_valid(node_status)
                                      THEN COALESCE(json_extract(node_status, '$.account'), ?)
                                      ELSE ? END = ?)
               ORDER BY id DESC
               LIMIT 1""",
        (trader_id, account, trader_id, account, account, account),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return store.row_to_checkpoint(row)


def has_prior_state(store: StatePersistence, account: str, trader_id: str) -> bool:
    """True if a LIVE start on ``account`` has saved risk state to carry.

    That is any trader's row carrying the account key, this trader's legacy row
    without the key, or a KEY ROTATION: this trader's newest row in the same venue
    + env (any key) is another key, whose risk and halt the restore carries
    (ae6ah, qa42q). The rotation read is the row :func:`latest_in_env`
    returns and the predicate is :func:`is_key_rotation`, the restore's own rule,
    so "the gate asks for consent" and "nothing carries" cannot drift apart
    (vibe-quant-neuvu.2). Rows of another trader whose ``node_status`` is not
    valid JSON are ignored: their account cannot be known, so counting them
    would let an unrelated row satisfy the gate. Never raises for row content.
    """
    cursor = store.conn.execute(
        """SELECT 1 FROM paper_trading_checkpoints
               WHERE (json_valid(node_status)
                      AND json_extract(node_status, '$.account') = ?)
                  OR (trader_id = ?
                      AND CASE WHEN json_valid(node_status)
                               THEN json_extract(node_status, '$.account') IS NULL
                               ELSE 0 END)
               LIMIT 1""",
        (account, trader_id),
    )
    if cursor.fetchone() is not None:
        return True
    prefix = account_env(account) + ":"
    newest = store.conn.execute(
        f"""SELECT CASE WHEN json_valid(node_status)
                           THEN json_extract(node_status, '$.account') END AS account
               FROM paper_trading_checkpoints
               WHERE {_ENV_ROW_WHERE}
               ORDER BY id DESC
               LIMIT 1""",
        (trader_id, prefix, prefix),
    ).fetchone()
    if newest is None:
        return False
    newest_account = newest["account"]
    return isinstance(newest_account, str) and is_key_rotation(newest_account, account)
