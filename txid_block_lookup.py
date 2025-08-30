#!/usr/bin/env python3
"""
txid_block_lookup.py

Look up block height and block timestamp for a list of Bitcoin TXIDs.

- Works in a Jupyter notebook or as a standalone script.
- Supports Blockstream (default) and Mempool.space APIs (both Esplora-compatible).
- Handles retries, rate limiting, and resumable runs.
- Outputs a clean CSV with: txid, block_height, block_time_unix, block_time_utc_iso, block_hash, api_source, error

USAGE (in Jupyter):
-------------------
from txid_block_lookup import enrich_txids_with_block_info
enrich_txids_with_block_info(
    input_csv="bitfinex_tx_dataset.csv",
    txid_column="txid",
    output_csv="tx_block_info.csv",
    merged_output_csv="bitfinex_tx_dataset_enriched.csv",
    api_source="blockstream",  # or "mempool"
    rate_limit_per_sec=4,
)

USAGE (CLI):
------------
python txid_block_lookup.py \
  --input_csv bitfinex_tx_dataset.csv \
  --txid_column txid \
  --output_csv tx_block_info.csv \
  --merged_output_csv bitfinex_tx_dataset_enriched.csv \
  --api_source blockstream \
  --rate_limit_per_sec 4

Requirements:
- requests, pandas, python-dateutil (usually included in Anaconda)
"""

import os
import sys
import time
import math
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
import requests

# -----------------------------
# API CONFIG
# -----------------------------
API_BASES = {
    "blockstream": "https://blockstream.info/api",      # mainnet
    "mempool": "https://mempool.space/api",             # mainnet
    # Add other Esplora endpoints if desired
}

# -----------------------------
# HELPERS
# -----------------------------
def iso_utc(ts_unix: Optional[int]) -> Optional[str]:
    if ts_unix is None or pd.isna(ts_unix):
        return None
    try:
        return datetime.fromtimestamp(int(ts_unix), tz=timezone.utc).isoformat()
    except Exception:
        return None

def unique_txids_from_csv(path: str, txid_column: str) -> pd.Series:
    df = pd.read_csv(path)
    if txid_column not in df.columns:
        raise ValueError(f"Column '{txid_column}' not found. Columns: {list(df.columns)}")
    txids = (
        df[txid_column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .drop_duplicates()
    )
    return txids

def load_existing_results(output_csv: str) -> pd.DataFrame:
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            return pd.read_csv(output_csv, dtype={"txid": str})
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "txid", "block_height", "block_time_unix", "block_time_utc_iso", "block_hash", "api_source", "error"
    ])

def save_results_incremental(df_partial: pd.DataFrame, output_csv: str):
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    df_partial.to_csv(output_csv, mode="a", index=False, header=write_header)

def rate_limit_sleep(last_call_ts: float, rate_limit_per_sec: float):
    if rate_limit_per_sec <= 0:
        return
    min_interval = 1.0 / rate_limit_per_sec
    elapsed = time.time() - last_call_ts
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

# -----------------------------
# FETCH FUNCTIONS
# -----------------------------
def fetch_tx_info_esplora(api_base: str, txid: str, session: requests.Session, timeout: int = 15) -> Dict:
    """
    Works for Esplora-compatible endpoints (Blockstream, Mempool.space).
    GET /api/tx/{txid}
    """
    url = f"{api_base}/tx/{txid}"
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def extract_block_fields_from_esplora(tx_json: Dict) -> Dict:
    """
    Esplora tx JSON includes a "status" object with:
      - confirmed (bool)
      - block_height (int)
      - block_hash (str)
      - block_time (unix int)
    """
    status = tx_json.get("status", {}) if isinstance(tx_json, dict) else {}
    if not status or not status.get("confirmed"):
        # Could be unconfirmed or unknown
        return {
            "block_height": None,
            "block_time_unix": None,
            "block_time_utc_iso": None,
            "block_hash": None
        }
    bh = status.get("block_height")
    bt = status.get("block_time")
    return {
        "block_height": bh,
        "block_time_unix": bt,
        "block_time_utc_iso": iso_utc(bt),
        "block_hash": status.get("block_hash")
    }

def robust_lookup(txid: str, api_source: str, session: requests.Session,
                  max_retries: int = 5, backoff_base: float = 0.8) -> Dict:
    """
    Retry with exponential backoff on transient errors (429, 5xx).
    """
    api_base = API_BASES[api_source]
    attempt = 0
    while True:
        try:
            tx_json = fetch_tx_info_esplora(api_base, txid, session=session)
            fields = extract_block_fields_from_esplora(tx_json)
            fields.update({"txid": txid, "api_source": api_source, "error": None})
            return fields
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            # 404: treat as final (maybe invalid or pruned)
            if status_code == 404:
                return {"txid": txid, "block_height": None, "block_time_unix": None,
                        "block_time_utc_iso": None, "block_hash": None,
                        "api_source": api_source, "error": "404 Not Found"}
            # 400-level (except 429) also likely final
            if status_code and 400 <= status_code < 500 and status_code != 429:
                return {"txid": txid, "block_height": None, "block_time_unix": None,
                        "block_time_utc_iso": None, "block_hash": None,
                        "api_source": api_source, "error": f"HTTP {status_code}"}
            # Otherwise retry
            attempt += 1
            if attempt > max_retries:
                return {"txid": txid, "block_height": None, "block_time_unix": None,
                        "block_time_utc_iso": None, "block_hash": None,
                        "api_source": api_source, "error": f"HTTP error after {max_retries} retries"}
            sleep_s = (backoff_base ** attempt) * (1.5 + (attempt % 3) * 0.2)
            time.sleep(sleep_s)
        except (requests.ConnectionError, requests.Timeout) as e:
            attempt += 1
            if attempt > max_retries:
                return {"txid": txid, "block_height": None, "block_time_unix": None,
                        "block_time_utc_iso": None, "block_hash": None,
                        "api_source": api_source, "error": f"Network/Timeout after {max_retries} retries"}
            sleep_s = (backoff_base ** attempt) * (1.5 + (attempt % 3) * 0.2)
            time.sleep(sleep_s)
        except Exception as e:
            # Unknown error: don't spin forever
            return {"txid": txid, "block_height": None, "block_time_unix": None,
                    "block_time_utc_iso": None, "block_hash": None,
                    "api_source": api_source, "error": f"Unexpected: {type(e).__name__}: {e}"}

# -----------------------------
# MAIN ENRICHMENT FUNCTION
# -----------------------------
def enrich_txids_with_block_info(
    input_csv: str,
    txid_column: str = "txid",
    output_csv: str = "tx_block_info.csv",
    merged_output_csv: Optional[str] = None,
    api_source: str = "blockstream",    # or "mempool"
    rate_limit_per_sec: float = 4.0,    # polite default; tune as needed
    checkpoint_every: int = 100
) -> pd.DataFrame:
    """
    Reads TXIDs from input_csv, queries API, writes/updates output_csv incrementally.
    If merged_output_csv is provided, also writes an enriched version joined back to the input CSV.
    Returns the final dataframe of results.
    """
    api_source = api_source.lower().strip()
    if api_source not in API_BASES:
        raise ValueError(f"api_source must be one of {list(API_BASES.keys())}")

    # Load txids
    txids = unique_txids_from_csv(input_csv, txid_column=txid_column)
    print(f"Found {len(txids)} unique TXIDs in '{input_csv}'.")

    # Load existing (for resume)
    existing = load_existing_results(output_csv)
    have = set(existing["txid"].astype(str).str.lower()) if not existing.empty else set()

    # Filter remaining
    to_fetch = [t for t in txids if t not in have]
    print(f"{len(to_fetch)} TXIDs to fetch (skipping {len(have)} already done).")

    session = requests.Session()
    results_batch = []
    last_call_ts = 0.0

    try:
        for i, txid in enumerate(to_fetch, 1):
            # Rate limit
            rate_limit_sleep(last_call_ts, rate_limit_per_sec)
            last_call_ts = time.time()

            rec = robust_lookup(txid, api_source=api_source, session=session)
            results_batch.append(rec)

            # Periodic checkpoint
            if (i % checkpoint_every == 0) or (i == len(to_fetch)):
                df_partial = pd.DataFrame(results_batch)
                save_results_incremental(df_partial, output_csv=output_csv)
                print(f"Checkpoint: wrote {len(df_partial)} rows to '{output_csv}' (i={i}/{len(to_fetch)}).")
                results_batch.clear()
    finally:
        # Flush any remaining
        if results_batch:
            df_partial = pd.DataFrame(results_batch)
            save_results_incremental(df_partial, output_csv=output_csv)
            print(f"Final flush: wrote {len(df_partial)} rows to '{output_csv}'.")

    # Re-load full results
    final_df = load_existing_results(output_csv)

    # Optional merge back to the original CSV
    if merged_output_csv:
        src = pd.read_csv(input_csv)
        merged = src.merge(final_df, how="left", left_on=txid_column, right_on="txid", suffixes=("", "_lookup"))
        merged.to_csv(merged_output_csv, index=False)
        print(f"Enriched dataset written to '{merged_output_csv}'.")

    return final_df

# -----------------------------
# CLI ENTRYPOINT
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Look up block height/time for Bitcoin TXIDs.")
    p.add_argument("--input_csv", required=True, help="Path to input CSV containing a TXID column")
    p.add_argument("--txid_column", default="txid", help="Name of the TXID column (default: txid)")
    p.add_argument("--output_csv", default="tx_block_info.csv", help="Where to append/write lookup results")
    p.add_argument("--merged_output_csv", default=None, help="Optional: write original CSV enriched with block info")
    p.add_argument("--api_source", default="blockstream", choices=list(API_BASES.keys()), help="API source to use")
    p.add_argument("--rate_limit_per_sec", type=float, default=4.0, help="Max requests per second")
    p.add_argument("--checkpoint_every", type=int, default=100, help="Write partial results every N lookups")
    return p.parse_args()

if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        # Running inside a Jupyter cell: do nothing here.
        pass
    else:
        args = parse_args()
        enrich_txids_with_block_info(
            input_csv=args.input_csv,
            txid_column=args.txid_column,
            output_csv=args.output_csv,
            merged_output_csv=args.merged_output_csv,
            api_source=args.api_source,
            rate_limit_per_sec=args.rate_limit_per_sec,
            checkpoint_every=args.checkpoint_every,
        )
