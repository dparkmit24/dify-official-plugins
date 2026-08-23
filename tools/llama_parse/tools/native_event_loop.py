"""Run LlamaParse's asyncio work on a native OS thread.

The Dify plugin runtime is gevent monkey-patched: every tool invocation runs
in a greenlet and all greenlets are multiplexed on one OS thread. asyncio
tracks its running event loop per OS thread, so two invocations driving event
loops concurrently corrupt each other's async state (RuntimeError "Detected
nested async...", anyio "cannot create weak reference to 'NoneType' object").
Running each parse on its own native thread gives it a private event loop.

The thread is spawned with the original ``_thread.start_new_thread`` saved by
gevent. ``monkey.get_original("threading", "Thread")`` is not enough: the
unpatched ``Thread`` class still resolves ``_start_new_thread`` through the
patched ``threading`` module globals and would spawn a greenlet instead.
"""

import asyncio
import time

try:
    from gevent import monkey

    _start_native_thread = monkey.get_original("_thread", "start_new_thread")
except ImportError:  # pragma: no cover - gevent always ships with dify_plugin
    from _thread import start_new_thread as _start_native_thread

_JOIN_POLL_SECONDS = 0.1


def run_async(coro_factory):
    """Run ``coro_factory()`` to completion on a private event loop in a
    native OS thread and return its result.

    The calling greenlet waits with the (gevent-patched) ``time.sleep``, so
    other invocations and the runtime's heartbeat keep running meanwhile.
    """
    outcome = {}

    def _worker():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                outcome["result"] = loop.run_until_complete(coro_factory())
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            outcome["error"] = exc
        finally:
            outcome["done"] = True

    _start_native_thread(_worker, ())
    while "done" not in outcome:
        time.sleep(_JOIN_POLL_SECONDS)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]
