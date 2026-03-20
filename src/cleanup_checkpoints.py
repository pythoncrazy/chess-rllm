"""List and optionally delete all Tinker checkpoints.

Usage:
    uv run python scripts/cleanup_checkpoints.py           # dry-run list
    uv run python scripts/cleanup_checkpoints.py --delete  # delete all
    uv run python scripts/cleanup_checkpoints.py --delete --keep-latest 1  # keep N newest per run
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import tinker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Actually delete (default: dry-run)")
    parser.add_argument("--keep-latest", type=int, default=0, metavar="N",
                        help="Keep N most-recent checkpoints per training run (0 = delete all)")
    args = parser.parse_args()

    sc = tinker.ServiceClient(base_url=os.environ.get("TINKER_BASE_URL"))
    rc = sc.create_rest_client()

    print("Fetching checkpoints...")
    all_ckpts = []
    offset = 0
    limit = 100
    while True:
        resp = rc.list_user_checkpoints(limit=limit, offset=offset).result()
        all_ckpts.extend(resp.checkpoints)
        total = resp.cursor.total_count
        offset += limit
        print(f"  {len(all_ckpts):>5} / {total}", end="\r")
        if offset >= total:
            break
    print(f"\nTotal checkpoints: {len(all_ckpts)}")

    # Group by training_run_id for --keep-latest
    by_run: dict[str, list] = defaultdict(list)
    for ck in all_ckpts:
        run_id = ck.tinker_path.split(":")[0].replace("tinker://", "")
        by_run[run_id].append(ck)

    to_delete = []
    if args.keep_latest > 0:
        for run_id, ckpts in by_run.items():
            sorted_ckpts = sorted(ckpts, key=lambda c: c.time, reverse=True)
            to_delete.extend(sorted_ckpts[args.keep_latest:])
        print(f"Keeping {args.keep_latest} newest per run → deleting {len(to_delete)} checkpoints")
    else:
        to_delete = all_ckpts

    total_gb = sum(c.size_bytes for c in to_delete) / 1e9
    print(f"\nCheckpoints to delete: {len(to_delete)}  ({total_gb:.1f} GB)")
    for ck in sorted(to_delete, key=lambda c: c.time):
        print(f"  {ck.time.strftime('%Y-%m-%d %H:%M')}  {ck.tinker_path}")

    if not args.delete:
        print("\n[Dry-run] Pass --delete to actually delete.")
        return

    confirm = input(f"\nDelete {len(to_delete)} checkpoints ({total_gb:.1f} GB)? [y/N] ")
    if confirm.lower() != "y":
        print("Aborted.")
        return

    deleted = 0
    failed = 0
    for ck in to_delete:
        try:
            rc.delete_checkpoint_from_tinker_path(ck.tinker_path).result()
            deleted += 1
            print(f"  deleted ({deleted}/{len(to_delete)})  {ck.tinker_path}", end="\r")
        except Exception as e:
            print(f"\n  FAILED {ck.tinker_path}: {e}")
            failed += 1

    print(f"\nDone. Deleted {deleted}, failed {failed}.")


if __name__ == "__main__":
    main()
