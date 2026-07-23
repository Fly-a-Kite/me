from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def create_safe_executor(
    executor_cls: type,
    *,
    max_workers: int,
    max_tasks_per_child: int | None = None,
) -> Any:
    """Create process pools with spawn so threaded parents never call fork()."""

    kwargs: dict[str, Any] = {"max_workers": max(1, int(max_workers))}
    try:
        is_process_pool = issubclass(executor_cls, ProcessPoolExecutor)
    except TypeError:
        is_process_pool = False
    if is_process_pool:
        kwargs["mp_context"] = multiprocessing.get_context("spawn")
        if max_tasks_per_child is not None:
            kwargs["max_tasks_per_child"] = max(1, int(max_tasks_per_child))
    return executor_cls(**kwargs)
