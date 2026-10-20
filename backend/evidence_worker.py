"""Read/write graph evidence in the configured authoring Python environment."""

import json
import sys


def main():
    from evo.contrib import catalog
    from evo.workbench import Workbench

    request = json.load(sys.stdin)
    wb = Workbench(request["root"], directory=request["directory"])
    method = request["method"]
    if method in ("models", "wishlist"):
        result = getattr(wb, method)
    elif method == "catalog":
        result = catalog()
    elif method in (
        "performance_records",
        "fixtures",
        "details",
        "add",
        "remove",
        "save_performance",
        "save_wish",
        "compare",
    ):
        result = getattr(wb, method)(*request.get("arguments", []))
    else:
        raise ValueError("Unknown evidence operation")
    print("KITE_EVIDENCE " + json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
