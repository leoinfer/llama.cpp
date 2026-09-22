#!/usr/bin/env python3
"""Fetch and classify llama.cpp changes relevant to Flash-Next on AMD Vulkan."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "ggml-org/llama.cpp"
QUERIES = (
    "qwen4exp",
    "Qwen3.8-Flash-Next",
    "GatedDeltaNet Vulkan",
    "hybrid cache Vulkan",
    "gfx1200 Vulkan",
    "RDNA4 Vulkan",
    "MoE Vulkan",
    "MTP speculative",
    "Flash Attention Vulkan",
    "cooperative matrix Vulkan",
)
RELEVANT = (
    "qwen4exp",
    "qwen3.8",
    "flash-next",
    "gateddeltanet",
    "gated delta",
    "hybrid",
    "qsa",
    "mtp",
    "speculative",
    "vulkan",
    "radv",
    "gfx1200",
    "rdna4",
    "moe",
    "quant",
    "flash attention",
    "cooperative matrix",
    "integer dot",
    "command buffer",
    "descriptor",
)


def _github_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "leo-llama-upstream-monitor/1",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected GitHub response from {url}")
    return value


def _head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _classify(item: dict) -> str:
    text = " ".join(
        str(item.get(key, ""))
        for key in ("title", "body", "labels")
    ).lower()
    return "RELEVANT" if any(term in text for term in RELEVANT) else "MAYBE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 100:
        parser.error("--limit must be between 1 and 100")

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    items: dict[str, dict[str, dict]] = {}
    failures: dict[str, str] = {}
    for term in QUERIES:
        query = urllib.parse.urlencode(
            {"q": f"repo:{REPO} {term}", "sort": "updated", "order": "desc", "per_page": args.limit}
        )
        try:
            payload = _github_json(f"https://api.github.com/search/issues?{query}")
            rows = payload.get("items", [])
            items[term] = {
                str(row.get("html_url")): {
                    "number": row.get("number"),
                    "title": row.get("title"),
                    "state": row.get("state"),
                    "updated_at": row.get("updated_at"),
                    "classification": _classify(row),
                    "labels": [label.get("name") for label in row.get("labels", [])],
                }
                for row in rows
                if isinstance(row, dict) and row.get("html_url")
            }
        except (OSError, ValueError, RuntimeError) as exc:
            failures[term] = str(exc)

    merged: dict[str, dict] = {}
    for term, term_rows in items.items():
        for url, row in term_rows.items():
            existing = merged.setdefault(url, {**row, "queries": []})
            existing["queries"].append(term)
    result = {
        "schema": "leo-llama-upstream-monitor.v1",
        "repo": REPO,
        "fetched_at": now,
        "local_head": _head(args.repo),
        "queries": list(QUERIES),
        "items": sorted(merged.values(), key=lambda row: row.get("updated_at") or "", reverse=True),
        "failures": failures,
        "policy": {
            "RELEVANT": "benchmark against the fork before merging",
            "MAYBE": "inspect only when a baseline bottleneck matches",
            "IGNORE": "do not carry into the local patch stack",
        },
    }
    output = args.output
    if output is None:
        output = args.repo.parent / "overnight-flash-next-campaign" / "03_BASELINE" / "LLAMA_UPSTREAM_MONITOR.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "items": len(result["items"]), "failures": len(failures)}))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
