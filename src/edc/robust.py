"""Watchdog for network work: run one function in a child process and kill it if it takes too long.

Why a process and not a thread: a thread that is stuck inside a GDAL download cannot be stopped in
Python. A child process can always be killed (TerminateProcess on Windows). Used by
scripts/03a_download_archive.py (one scene = one child process).
"""
import multiprocessing as mp
import os
import time


def _child(conn, func, args, stall_s):
    """Runs in the child: func(*args) -> send ("ok", result, seconds) or ("error", message)."""
    t0 = time.perf_counter()
    try:
        if stall_s:                       # stall test only: behave like a frozen download
            time.sleep(stall_s)
        conn.send(("ok", func(*args), round(time.perf_counter() - t0, 3)))
    except BaseException as e:
        conn.send(("error", f"{type(e).__name__}: {str(e)[:300]}"))
    conn.close()
    os._exit(0)   # GDAL network threads can keep a process alive at exit; the result is already sent


def run_in_child(func, args, timeout_s, stall_s=0):
    """Run func(*args) in a child process; kill it after timeout_s.

    func must be importable (a top-level function in a module), and args/result picklable.
    Returns (result, child_seconds). Raises TimeoutError or RuntimeError, so the caller can retry.
    """
    ctx = mp.get_context("spawn")                     # Windows only supports "spawn"
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(child, func, args, stall_s), daemon=True)
    p.start()
    child.close()
    try:
        if not parent.poll(timeout_s):                # nothing back in time -> frozen
            raise TimeoutError(f"did not finish within {timeout_s:.0f} s; child process killed")
        msg = parent.recv()
    except EOFError:                                  # child died without sending anything
        raise RuntimeError(f"child process died (exit code {p.exitcode})")
    finally:
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join(10)
    if msg[0] != "ok":
        raise RuntimeError(msg[1])
    return msg[1], msg[2]
