"""Entry point: ``uv run python -m bhel_optimize.server``."""

from bhel_optimize.server.app import main

if __name__ == "__main__":
    raise SystemExit(main())
