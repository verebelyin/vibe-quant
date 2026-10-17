/**
 * Symbol choice list for the Data page ingest form (bead vibe-quant-yi8sl).
 *
 * The form must offer every symbol the downloader can ingest
 * (``GET /api/data/symbols/supported``) so a brand-new coin's first ingest can
 * be started from the UI, and mark the ones that already have archived data
 * (``GET /api/data/symbols``). Supported order wins; archive-only symbols
 * (e.g. delisted or historically-ingested extras) are appended after it.
 */

export const HAS_DATA_LABEL = "has data";

export interface SymbolOption {
  symbol: string;
  hasData: boolean;
}

export function buildSymbolOptions(supported: string[], archived: string[]): SymbolOption[] {
  const archivedSet = new Set(archived);
  const seen = new Set<string>();
  const options: SymbolOption[] = [];

  for (const symbol of supported) {
    if (seen.has(symbol)) continue;
    seen.add(symbol);
    options.push({ symbol, hasData: archivedSet.has(symbol) });
  }
  for (const symbol of archived) {
    if (seen.has(symbol)) continue;
    seen.add(symbol);
    options.push({ symbol, hasData: true });
  }

  return options;
}
