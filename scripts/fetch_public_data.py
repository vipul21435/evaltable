"""Download the public result files used for the numbers in the README.

Nothing fetched here is committed to this repository.

* SWE-bench/experiments (no licence file in that repository): ``results/results.json`` of every
  submission under ``evaluation/<split>/``, at a pinned commit.
* UKGovernmentBEIS/inspect_ai (MIT licence): the Inspect evaluation logs kept in its test
  suite, at a pinned commit.

Usage: python scripts/fetch_public_data.py OUT_DIR [--split verified]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SWEBENCH_REPO = "SWE-bench/experiments"
SWEBENCH_COMMIT = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
INSPECT_REPO = "UKGovernmentBEIS/inspect_ai"
INSPECT_COMMIT = "e4435baa530d05324358bedc3782609c952f9876"
INSPECT_LOGS = [
    "tests/scorer/logs/2025-02-11T15-17-00-05-00_popularity_dPiJifoWeEQBrfWsAopzWr.eval",
    "tests/scorer/logs/2025-02-11T15-18-04-05-00_popularity_mj7khqpMM4GBCfVQozKgzB.eval",
    "tests/analysis/test_logs/2025-05-12T20-28-13-04-00_popularity.json",
    "tests/analysis/test_logs/2025-05-12T20-28-26-04-00_security-guide.json",
]


def _get(url: str, retries: int = 3) -> bytes:
    headers = {"User-Agent": "evaltable-fetch"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def raw_url(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def fetch_swebench(out: Path, split: str) -> int:
    api = f"https://api.github.com/repos/{SWEBENCH_REPO}/contents/evaluation/{split}"
    listing = json.loads(_get(f"{api}?ref={SWEBENCH_COMMIT}"))
    names = sorted(entry["name"] for entry in listing if entry["type"] == "dir")
    count = 0
    for name in names:
        rel = f"evaluation/{split}/{name}/results/results.json"
        target = out / "swebench" / rel
        if target.exists():
            count += 1
            continue
        try:
            data = _get(raw_url(SWEBENCH_REPO, SWEBENCH_COMMIT, rel))
        except urllib.error.HTTPError as exc:
            print(f"skip {name}: HTTP {exc.code}", file=sys.stderr)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        count += 1
    return count


def fetch_inspect(out: Path) -> int:
    for rel in INSPECT_LOGS:
        target = out / "inspect" / Path(rel).name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_get(raw_url(INSPECT_REPO, INSPECT_COMMIT, rel)))
    return len(INSPECT_LOGS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="directory to download into")
    parser.add_argument("--split", default="verified", help="SWE-bench split (default: verified)")
    args = parser.parse_args(argv)
    n_swe = fetch_swebench(args.out, args.split)
    n_inspect = fetch_inspect(args.out)
    print(f"SWE-bench {args.split}: {n_swe} results.json files at {SWEBENCH_COMMIT[:12]}")
    print(f"Inspect: {n_inspect} logs at {INSPECT_COMMIT[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
