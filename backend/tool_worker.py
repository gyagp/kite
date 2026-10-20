"""Run Model Lab commands with Kite's user-facing command name."""

import sys

from evo.cli import main

if __name__ == "__main__":
    sys.argv[0] = "kite" + (" " + sys.argv[1] if len(sys.argv) > 1 else "")
    raise SystemExit(main(prog="kite"))
