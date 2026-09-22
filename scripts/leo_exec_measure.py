#!/usr/bin/env python3
"""Run one command and emit elapsed/max-RSS resource data on stderr."""
from __future__ import annotations

import json
import resource
import subprocess
import sys
import time


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: leo_exec_measure.py COMMAND [ARG ...]", file=sys.stderr)
        return 2
    started = time.perf_counter_ns()
    completed = subprocess.run(sys.argv[1:], check=False)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    payload = {
        "command": sys.argv[1:],
        "returncode": completed.returncode,
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "child_user_s": usage.ru_utime,
        "child_system_s": usage.ru_stime,
        "child_max_rss_kib": usage.ru_maxrss,
        "platform": "linux-kib-maxrss",
    }
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
