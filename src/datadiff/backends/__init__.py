from __future__ import annotations

import importlib

from datadiff.backends.base import Backend


def make_backend(name: str) -> Backend:
    targets_module = importlib.import_module("datadiff." + "targets")
    instantiate = getattr(targets_module, "instantiate_target_backend")
    return instantiate(name)
