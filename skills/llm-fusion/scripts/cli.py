"""Portable CLI entry point for llm-fusion.

All runtime code lives under ``scripts/`` so the skill can run without a
separate installable Python package.
"""

import argparse
import json


VERSION = "0.2.9"


def build_parser():
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-fusion",
        description="Multi-model fusion pipeline — ensemble multiple LLM responses.",
    )
    parser.add_argument(
        "--query", "-q",
        help="User query to process through the fusion pipeline.",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to fusion_config.yaml. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory for saved fusion JSON results.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information to stderr.",
    )
    parser.add_argument(
        "--tier", "-t",
        default="medium",
        help="Panel model tier: low1 (2 calls: 1 deepseek + 1 mimo), low2 (4 calls: 2 deepseek + 2 mimo), low3 (6 calls: 3 deepseek + 3 mimo), medium (3 calls: deepseek + mimo + deepseek-v4-pro, default), high (3 calls: deepseek-v4-pro + minimax-m3 + qwen3.7-plus).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and arguments without making API calls.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit.",
    )
    return parser


def main(argv=None):
    """CLI entry point. Return exit code (0 for success, 2 for error)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"llm-fusion {VERSION}")
        return 0

    if args.config:
        resolved_config = args.config
    else:
        from scripts.config import _discover_config_path
        resolved_config = _discover_config_path()

    if args.dry_run:
        info = {
            "ok": True,
            "query": args.query,
            "config": resolved_config,
            "output_dir": args.output_dir,
            "verbose": args.verbose,
            "tier": args.tier,
        }
        print(json.dumps(info, indent=2))
        return 0

    if not args.query:
        parser.error("--query is required (use --dry-run to validate without a query)")

    from scripts.pipeline import run_pipeline
    from scripts.output import format_for_chat

    result = run_pipeline(
        args.query,
        config_path=args.config,
        output_dir=args.output_dir,
        verbose=args.verbose,
        tier=args.tier,
    )

    output = format_for_chat(result, include_metadata=True)
    print(output)

    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
