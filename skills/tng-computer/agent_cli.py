"""One-shot TNG computer CLI.

Usage (from the repo root):

    python skills/tng-computer/agent_cli.py "where is Commander Data"
    python skills/tng-computer/agent_cli.py "shield status?" --episode 100161.txt
    python skills/tng-computer/agent_cli.py "where is Captain Picard?" --backend openai

The default ``extractive`` backend is offline (state + golden memory) and needs
no API key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agent import TNGComputerAgent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer as the Enterprise computer.")
    parser.add_argument("query", nargs="+", help="the crew's question/command")
    parser.add_argument("--episode", default=None, help="episode file, e.g. 100101.txt")
    parser.add_argument("--scene", default=None, help="scene, e.g. Bridge")
    parser.add_argument("--backend", default="extractive", choices=["extractive", "openai"])
    parser.add_argument("--no-state", action="store_true", help="drop the SHIP STATE block")
    parser.add_argument("--no-context", action="store_true", help="drop the EPISODE CONTEXT block")
    args = parser.parse_args()

    agent = TNGComputerAgent(backend=args.backend)
    query = " ".join(args.query)
    print(agent.respond(
        query,
        episode=args.episode,
        scene=args.scene,
        use_state=not args.no_state,
        use_context=not args.no_context,
    ))


if __name__ == "__main__":
    main()