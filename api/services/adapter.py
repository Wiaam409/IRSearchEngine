from typing import TypeVar, Callable, Any
from fastapi.concurrency import run_in_threadpool

T = TypeVar("T")

async def execute_sync_service(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Safely executes synchronous, CPU-bound IR code in FastAPI's background thread pool
    to avoid blocking the async event loop.
    """
    def wrapper():
        return func(*args, **kwargs)
    return await run_in_threadpool(wrapper)
