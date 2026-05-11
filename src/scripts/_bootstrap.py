from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ensure_local_package(package_name: str = "bruce_code") -> None:
    if package_name in sys.modules:
        return

    root = Path(__file__).resolve().parents[1]
    init_file = root / "__init__.py"
    if not init_file.exists():
        raise FileNotFoundError(f"Package entry not found: {init_file}")

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to bootstrap local package {package_name!r} from {root}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
