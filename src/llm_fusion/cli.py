"""Portable CLI entry point for llm-fusion.

Provides the canonical argument parser and main() function.
Both the installed ``llm-fusion`` console script and the skill
wrapper script delegate here for consistent behaviour.
"""

import argparse
import json
import sys


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
    """CLI entry point.  Return exit code (0 for success, 2 for error)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from llm_fusion import __version__
        print(f"llm-fusion {__version__}")
        return 0

    # Resolve config path early for dry-run
    if args.config:
        resolved_config = args.config
    else:
        from llm_fusion.config import _discover_config_path
        resolved_config = _discover_config_path()

    if args.dry_run:
        info = {
            "ok": True,
            "query": args.query,
            "config": resolved_config,
            "output_dir": args.output_dir,
            "verbose": args.verbose,
        }
        print(json.dumps(info, indent=2))
        return 0

    if not args.query:
        parser.error("--query is required (use --dry-run to validate without a query)")

    from llm_fusion.pipeline import run_pipeline
    from llm_fusion.output import format_for_chat

    result = run_pipeline(
        args.query,
        config_path=args.config,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    output = format_for_chat(result, include_metadata=True)
    print(output)

    return 0 if result.get("success") else 2
