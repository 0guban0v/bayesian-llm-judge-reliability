"""Run a profiling command and persist a small summary JSON for comparisons."""

from __future__ import annotations

import argparse
import json
import logging
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

LOGGER = logging.getLogger("profile_command")


def normalize_child_ru_maxrss_bytes(raw_ru_maxrss: int, platform: str) -> int:
    """Normalize child-process ru_maxrss into bytes across platforms."""

    normalized = max(0, raw_ru_maxrss)
    if platform == "darwin":
        return normalized
    return normalized * 1024


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Logical profile target name.")
    parser.add_argument("--profile-path", type=Path, default=None, help="Speedscope output path.")
    parser.add_argument("--config", type=Path, required=True, help="Config path for this run.")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute after '--'.",
    )
    return parser.parse_args()


def build_metrics_path(
    target: str,
    started_at: datetime,
    profile_path: Path | None,
) -> Path:
    """Return the metrics JSON path for a profile or metrics-only run."""

    metrics_dir = profile_path.parent / "metrics" if profile_path is not None else Path("profiles") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if profile_path is not None:
        metrics_stem = profile_path.stem
    else:
        metrics_stem = f"{target}_{started_at.strftime('%Y%m%d_%H%M%S')}"
    return metrics_dir / f"{metrics_stem}.json"


def main() -> None:
    """Execute the requested command and persist timing metadata."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [profile_command] %(message)s",
    )
    args = parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("No command provided to profile.")

    started_at = datetime.now(UTC)
    profile_path = args.profile_path.resolve() if args.profile_path is not None else None
    if profile_path is not None:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = build_metrics_path(args.target, started_at, profile_path)
    started_perf = time.perf_counter()
    completed = subprocess.run(args.command, check=False)
    wall_time_sec = time.perf_counter() - started_perf
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss_bytes = normalize_child_ru_maxrss_bytes(child_usage.ru_maxrss, sys.platform)
    payload = {
        "target": args.target,
        "started_at_utc": started_at.isoformat(),
        "config": str(args.config.resolve()),
        "profile_path": str(profile_path) if profile_path is not None else None,
        "metrics_path": str(metrics_path),
        "wall_time_sec": wall_time_sec,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_mb": peak_rss_bytes / (1024 * 1024),
        "exit_code": completed.returncode,
        "command": args.command,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote metrics %s", metrics_path)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
