"""List all Tinker checkpoints grouped by training run.

Usage:
    uv run python scripts/list_checkpoints.py
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import tinker


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main() -> None:
    sc = tinker.ServiceClient(base_url=os.environ.get("TINKER_BASE_URL"))
    rc = sc.create_rest_client()

    print("Fetching checkpoints...")
    all_ckpts = []
    offset = 0
    while True:
        resp = rc.list_user_checkpoints(limit=100, offset=offset).result()
        all_ckpts.extend(resp.checkpoints)
        total = resp.cursor.total_count
        offset += 100
        print(f"  {len(all_ckpts):>5} / {total}", end="\r")
        if offset >= total:
            break
    print()

    by_run: dict[str, list] = defaultdict(list)
    for ck in all_ckpts:
        run_id = ck.tinker_path.split(":")[0].replace("tinker://", "")
        by_run[run_id].append(ck)

    runs_sorted = sorted(
        by_run.items(),
        key=lambda kv: max(c.time for c in kv[1]),
    )

    grand_total_bytes = 0
    grand_total_count = 0

    for run_id, ckpts in runs_sorted:
        total_bytes = sum(c.size_bytes for c in ckpts)
        grand_total_bytes += total_bytes
        grand_total_count += len(ckpts)
        newest = max(c.time for c in ckpts)
        latest_step = max((c.checkpoint_id for c in ckpts), key=lambda s: s.split("/")[-1])
        print(
            f"{run_id[:8]}  {newest.strftime('%Y-%m-%d %H:%M')}  "
            f"{len(ckpts):>5} ckpts  {fmt_bytes(total_bytes):>10}  latest={latest_step}"
        )

    print(f"\nTotal: {grand_total_count} checkpoints  {fmt_bytes(grand_total_bytes)}")


if __name__ == "__main__":
    main()
